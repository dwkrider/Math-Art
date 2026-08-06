
# Topological Surface Generator for Blender
#
# The classic topology menagerie, after chapter 6 of Henry Segerman's
# "Visualizing Mathematics with 3D Printing" (figs 6-1..6-7) plus the
# Klein bottle from the project TODO:
#
#   * Klein bottle -- the iconic "bottle" shape (the standard smooth
#     closed-form immersion) and the figure-8 / twisted-torus form.
#   * Cross-cap and Steiner's Roman surface -- the two classical
#     immersions of the real projective plane RP^2.
#   * Boy's surface via the Bryant-Kusner parametrization (an RP^2
#     immersion with no pinch points).
#   * Orientable genus-g handlebody surfaces, meshed implicitly with
#     the marching-tetrahedra kernel from minimal_surface_toolkit.
#   * Solid closed strips with n half-twists (n = 1 is a printable
#     Mobius band, fig 6-1), swept as a watertight solid directly.
#   * The Sudanese Mobius band -- Lawson's minimal Mobius band in S^3,
#     stereographically projected to R^3 (embedded, with a round
#     great-circle boundary).
#
# Non-orientable surfaces cannot embed in 3-space, so KLEIN / KLEIN8 /
# CROSSCAP / ROMAN / BOY are immersions with self-intersections. The
# parametric grids are closed combinatorially -- boundary
# identifications are made by vertex index, not by coordinate welding
# -- so each mesh carries the Euler characteristic of the abstract
# surface: 0 for the Klein bottles, 1 for the RP^2 models. Parameter
# grids are offset by fractional steps where needed so that no two
# grid samples land exactly on a self-intersection curve.
#
# References:
#   Klein bottle: F. Klein (1882). Boy's surface: W. Boy, Math. Ann.
#       57 (1903), here via the R. Bryant - R. Kusner parametrization.
#   Cross-cap and Roman surface: two immersions of RP^2 due to
#       J. Steiner (Rome, 1844). Mobius band: A. F. Mobius (1858).
#   Sudanese Mobius band: H. B. Lawson, "Complete Minimal Surfaces in
#       S^3", Ann. of Math. 92 (1970), 335-374; named for Sue Goodman
#       and Daniel Asimov (cf. G. Francis, "A Topological Picturebook",
#       Springer 1987).
#   Menagerie after ch. 6 of H. Segerman, "Visualizing Mathematics
#       with 3D Printing" (2016).

bl_info = {
    "name": "Topological Surface Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Topological Surface",
    "description": "Klein bottles, cross-cap, Roman and Boy surfaces, "
                   "genus-g handlebodies, solid twisted strips",
    "category": "Add Mesh",
}

import math
import numpy as np

TAU = 2.0 * math.pi


# ==========================================================================
# Mesh bookkeeping helpers (used by the standalone self-tests too)
# ==========================================================================

def edge_face_counts(faces):
    """{sorted edge tuple: number of incident faces}."""
    cnt = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    return cnt


def euler_characteristic(nverts, faces):
    return nverts - len(edge_face_counts(faces)) + len(faces)


# ==========================================================================
# Klein bottles
# ==========================================================================

def build_klein_bottle(nu, nv):
    """The iconic bottle-shaped Klein immersion (the standard smooth
    closed-form parametrization, u in [0, pi], v in [0, 2pi]). The
    u = pi rim coincides with the u = 0 rim under v -> pi - v.  The
    seam is left SPLIT (coincident duplicate vertices, no index
    gluing): welding it makes the winding flip there, and averaged
    smooth normals then degenerate into a dark shading crease.  Split,
    each side shades smoothly and the renderer's double-sided normal
    flip hides the join.  Cut along that rim the surface is an
    orientable cylinder, so chi is still 0."""
    nv += nv % 2
    u = math.pi * np.arange(nu + 1)[:, None] / nu
    v = TAU * np.arange(nv)[None, :] / nv
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)
    x = (-2.0 / 15.0) * cu * (3 * cv - 30 * su + 90 * cu ** 4 * su
                              - 60 * cu ** 6 * su + 5 * cu * cv * su)
    y = (-1.0 / 15.0) * su * (3 * cv - 3 * cu ** 2 * cv
                              - 48 * cu ** 4 * cv + 48 * cu ** 6 * cv
                              - 60 * su + 5 * cu * cv * su
                              - 5 * cu ** 3 * cv * su
                              - 80 * cu ** 5 * cv * su
                              + 80 * cu ** 7 * cv * su)
    z = (2.0 / 15.0) * sv * (3 + 5 * cu * su)
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append((i * nv + j, i * nv + j2,
                          (i + 1) * nv + j2, (i + 1) * nv + j))
    return V, faces


def build_klein_figure8(nu, nv, radius=2.0):
    """Figure-8 (twisted-torus) Klein immersion: the cross-section is a
    figure-8 that makes a half-turn per revolution. The u = 2pi seam
    coincides with u = 0 under v -> -v but is left SPLIT (coincident
    duplicate vertices) -- see build_klein_bottle for why.  v samples
    sit at half-steps so no column lands on the figure-8 crossing
    point."""
    u = TAU * np.arange(nu + 1)[:, None] / nu
    v = TAU * (np.arange(nv)[None, :] + 0.5) / nv
    c2, s2 = np.cos(u / 2), np.sin(u / 2)
    sv, s2v = np.sin(v), np.sin(2 * v)
    r = radius + c2 * sv - s2 * s2v
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = s2 * sv + c2 * s2v
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append((i * nv + j, i * nv + j2,
                          (i + 1) * nv + j2, (i + 1) * nv + j))
    return V, faces


# ==========================================================================
# Sudanese Mobius band (Lawson's minimal Mobius band in S^3)
# ==========================================================================
# Half of Lawson's minimally-immersed Klein bottle in the unit 3-sphere
#   x(t, v) = (cos t cos v, sin t cos v, cos 2t sin v, sin 2t sin v),
# taking v in [0, pi], is an EMBEDDED Mobius band whose single boundary
# is a great circle.  Stereographically projecting to R^3 keeps that
# boundary a round circle.  We project from the point of S^3 farthest
# from the band, p = (-1, 0, -1, 0)/sqrt2 (the band's dot with p peaks
# at 1/sqrt2, so the denominator 1 - x.p never drops below ~0.293 and
# the image stays bounded).  In the basis e1 = (0,1,0,0),
# e2 = (0,0,0,1), e3 = (1,0,-1,0)/sqrt2 of p^perp this reduces to
#   X = x2/s,  Y = x4/s,  Z = (x1 - x3)/(sqrt2 s),  s = 1 + (x1+x3)/sqrt2.
# The nickname "Sudanese" honours topologists Sue Goodman and Daniel
# Asimov (Sue + Dan), not the country.

def build_sudanese_mobius(nu, nv):
    """Lawson's minimal Mobius band in S^3, stereographically projected
    to R^3.  The t = 0 and t = pi seam rows coincide in space (with a
    flip v -> pi - v), so -- exactly as for the Klein bottles above --
    the grid is left SPLIT there rather than index-glued: welding flips
    the winding and averaged smooth normals then form a dark crease.
    Cut open along the seam the mesh is a disk (chi 1) whose two ends
    meet on the boundary circle."""
    R2 = math.sqrt(2.0)
    t = math.pi * np.arange(nu + 1)[:, None] / nu       # around
    v = math.pi * np.arange(nv + 1)[None, :] / nv        # across
    ct, st = np.cos(t), np.sin(t)
    cv, sv = np.cos(v), np.sin(v)
    x1 = ct * cv
    x2 = st * cv
    x3 = np.cos(2 * t) * sv
    x4 = np.sin(2 * t) * sv
    s = 1.0 + (x1 + x3) / R2                              # = 1 - x . p
    x = x2 / s
    y = x4 / s
    z = (x1 - x3) / (R2 * s)
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    stride = nv + 1
    faces = []
    for i in range(nu):
        for j in range(nv):
            faces.append((i * stride + j, i * stride + j + 1,
                          (i + 1) * stride + j + 1, (i + 1) * stride + j))
    return V, faces


# ==========================================================================
# RP^2 immersions: cross-cap, Roman surface, Boy's surface
# ==========================================================================

def _rp2_quotient(nu, nv, fn, theta_offset):
    """Mesh a hemisphere parametrization fn(theta, phi) -> (x, y, z),
    phi in (0, pi/2], with the phi = pi/2 pole collapsed to one vertex
    and the phi = 0 equator glued to itself by theta -> theta + pi
    (the RP^2 quotient). nu must be even. Returns (verts, faces) with
    Euler characteristic 1 by construction."""
    half = nu // 2
    th = TAU * (np.arange(nu) + theta_offset) / nu
    x, y, z = fn(th[:1], math.pi / 2)
    verts = [np.array([x[0], y[0], z[0]])]
    for k in range(1, nv):
        ph = (math.pi / 2) * (1.0 - k / (nv - 1))
        m = nu if k < nv - 1 else half
        x, y, z = fn(th[:m], ph)
        verts.extend(np.stack(np.broadcast_arrays(x, y, z), axis=-1))

    def rid(k, j):
        j %= nu
        if k < nv - 1:
            return 1 + (k - 1) * nu + j
        return 1 + (nv - 2) * nu + (j % half)

    faces = [(0, rid(1, j + 1), rid(1, j)) for j in range(nu)]
    for k in range(1, nv - 1):
        for j in range(nu):
            faces.append((rid(k, j), rid(k, j + 1),
                          rid(k + 1, j + 1), rid(k + 1, j)))
    return np.array(verts), faces


def _crosscap_pt(th, ph):
    """Standard cross-cap immersion of RP^2; antipodes
    of the sphere (th, ph latitude) map to the same point."""
    st, ct = np.sin(th), np.cos(th)
    return (0.5 * st * np.sin(2 * ph) + 0 * th,
            0.5 * np.sin(2 * th) * np.cos(ph) ** 2,
            0.5 * np.cos(2 * th) * np.cos(ph) ** 2)


def _roman_pt(th, ph):
    """Steiner's Roman surface: the sphere mapped through
    (x, y, z) -> (yz, zx, xy)."""
    cp, sp = np.cos(ph), np.sin(ph)
    return (np.sin(th) * cp * sp,
            np.cos(th) * cp * sp,
            np.sin(th) * np.cos(th) * cp * cp)


def build_crosscap(nu, nv):
    nu += nu % 2               # theta -> theta + pi must be a grid map
    return _rp2_quotient(nu, nv, _crosscap_pt, 0.5)


def build_roman(nu, nv):
    nu += (-nu) % 4            # quarter-offset grid: need 4 | nu
    return _rp2_quotient(nu, nv, _roman_pt, 0.25)


def build_boy(ntheta, nrings):
    """Boy's surface via the Bryant-Kusner parametrization on the unit
    disk (polar grid), with the boundary circle glued antipodally
    (z ~ -z on |z| = 1) by vertex index. The three poles of the
    denominator inside the disk are the planar ends of the underlying
    minimal surface; they invert to the triple point at the origin, and
    samples landing on them are nudged off."""
    ntheta += ntheta % 2
    half = ntheta // 2
    th = TAU * (np.arange(ntheta) + 0.5) / ntheta
    s5 = math.sqrt(5.0)

    def bk(zc):
        w = zc ** 6 + s5 * zc ** 3 - 1
        bad = np.abs(w) < 1e-7
        if np.any(bad):
            zc = np.where(bad, zc * 1.01, zc)
            w = zc ** 6 + s5 * zc ** 3 - 1
        g1 = -1.5 * (zc * (1 - zc ** 4) / w).imag
        g2 = -1.5 * (zc * (1 + zc ** 4) / w).real
        g3 = ((1 + zc ** 6) / w).imag - 0.5
        s = g1 * g1 + g2 * g2 + g3 * g3
        return np.stack([g1 / s, g2 / s, g3 / s], axis=-1)

    verts = [bk(np.zeros(1, dtype=complex))[0]]
    for k in range(1, nrings + 1):
        r = k / nrings
        m = ntheta if k < nrings else half
        verts.extend(bk(r * np.exp(1j * th[:m])))

    def rid(k, j):
        j %= ntheta
        if k < nrings:
            return 1 + (k - 1) * ntheta + j
        return 1 + (nrings - 1) * ntheta + (j % half)

    faces = [(0, rid(1, j + 1), rid(1, j)) for j in range(ntheta)]
    for k in range(1, nrings):
        for j in range(ntheta):
            faces.append((rid(k, j), rid(k, j + 1),
                          rid(k + 1, j + 1), rid(k + 1, j)))
    return np.array(verts), faces


# ==========================================================================
# Genus-g handlebody surface (implicit, marching tetrahedra)
# ==========================================================================
# q(x, y) = product over g+1 overlapping circles in a row of the
# NORMALIZED factors (rho_i^2 - r^2) / (rho_i^2 + r^2) in (-1, 1),
# rho_i = distance to center i. Normalizing keeps distant circles'
# factors near +1 instead of growing without bound, so the width of
# the saddle channel where adjacent circle interiors connect (at the
# circle-circle crossing points q has a saddle of value 0) stays
# O(sqrt(eps)) independent of g -- with raw factors the channel
# shrinks below the marching-grid cell size for g >= 3 and the solid
# falls apart. The planar region {q <= eps} is then the blob around
# the circle union minus the g lens cores (inside exactly two circles
# the product is positive; its peak is ~0.034 even for the middle
# lens at g = 5, comfortably above eps). Adding k z^2 keeps every
# vertical fiber of the solid an interval, so {q + k z^2 <= eps} is a
# genus-g handlebody and its boundary has Euler char 2 - 2g.

_GENUS_R = 1.0          # circle radius
_GENUS_SPACING = 1.4    # center spacing (< 2r: adjacent circles overlap)
_GENUS_LEVEL = 0.015    # eps: below the lens-core peak for g = 1..5
_GENUS_ZK = 2.0         # z^2 coefficient: slab half-height <= ~0.42


def build_genus(genus, cell=0.125):
    try:
        from . import minimal_surface_toolkit as mst
    except ImportError:
        import minimal_surface_toolkit as mst
    g = genus
    r, d = _GENUS_R, _GENUS_SPACING
    cs = [(i - g / 2.0) * d for i in range(g + 1)]

    def field(x, y, z):
        q = np.ones_like(x)
        for c in cs:
            rho2 = (x - c) ** 2 + y ** 2
            q = q * (rho2 - r * r) / (rho2 + r * r)
        return q + _GENUS_ZK * z * z - _GENUS_LEVEL

    m = 0.6
    bmin = (cs[0] - r - m, -r - m, -0.55)
    bmax = (cs[-1] + r + m, r + m, 0.55)
    res = tuple(max(8, int(round((bmax[i] - bmin[i]) / cell)))
                for i in range(3))
    return mst.marching_tets(field, bmin, bmax, res)


# ==========================================================================
# Solid closed strip with n half-twists (Mobius band and friends)
# ==========================================================================

def build_twist_strip(half_twists, segments, width=0.6, thick=0.18,
                      ridge=False, radius=1.5):
    """Sweep a rectangular cross-section (optionally with a raised
    center-line ridge on both wide faces, as in Segerman fig 6-1)
    around a circle, turning it by n*pi over one revolution. The
    cross-section point list is symmetric under a half-turn (index
    shift k/2), so for odd n the seam closes with an index shift and
    the result is a single watertight solid -- printable directly."""
    n = half_twists
    m = max(segments, 8 * max(abs(n), 1))
    w2, t2 = width / 2.0, thick / 2.0
    if ridge:
        bw, bh = 0.16 * width, 0.7 * thick
        prof = [(w2, -t2), (w2, t2), (bw, t2), (0.0, t2 + bh),
                (-bw, t2), (-w2, t2), (-w2, -t2), (-bw, -t2),
                (0.0, -t2 - bh), (bw, -t2)]
    else:
        prof = [(w2, -t2), (w2, t2), (-w2, t2), (-w2, -t2)]
    k = len(prof)
    shift = (n % 2) * (k // 2)
    verts = []
    for j in range(m):
        t = TAU * j / m
        al = 0.5 * n * t
        ca, sa = math.cos(al), math.sin(al)
        ct, st = math.cos(t), math.sin(t)
        for (a, b) in prof:
            ar = a * ca - b * sa       # rotate in the (radial, z) plane
            br = a * sa + b * ca
            verts.append(((radius + ar) * ct, (radius + ar) * st, br))
    faces = []
    for j in range(m):
        j2 = (j + 1) % m
        s = shift if j == m - 1 else 0
        for i in range(k):
            i2 = (i + 1) % k
            faces.append((j * k + i, j * k + i2,
                          j2 * k + (i2 + s) % k, j2 * k + (i + s) % k))
    return np.array(verts), faces


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


PRESET_ITEMS = [
    ('KLEIN', "Klein Bottle",
     "Classical bottle-shaped Klein bottle immersion"),
    ('KLEIN8', "Klein Bottle (Figure-8)",
     "Figure-8 / twisted-torus Klein bottle immersion"),
    ('SUDANESE', "Sudanese Mobius Band",
     "Lawson's minimal Mobius band in S^3, stereographically "
     "projected to R^3 (embedded, round boundary circle)"),
    ('CROSSCAP', "Cross-Cap",
     "Standard cross-cap immersion of the projective plane"),
    ('ROMAN', "Roman Surface",
     "Steiner's Roman surface (projective plane)"),
    ('BOY', "Boy's Surface",
     "Boy's surface, Bryant-Kusner parametrization"),
    ('GENUS', "Genus-g Surface",
     "Orientable genus-g handlebody surface (implicit)"),
    ('TWIST_STRIP', "Twisted Strip (solid)",
     "Solid closed strip with n half-twists; n = 1 is a Mobius band"),
]

_IMMERSIONS = {'KLEIN', 'KLEIN8', 'SUDANESE', 'CROSSCAP', 'ROMAN', 'BOY'}


if _IN_BLENDER:

    def _new_object(context, name, verts, faces, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(map(float, v)) for v in np.asarray(verts)],
                       [], [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth',
                                [smooth] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    class MESH_OT_topological_surface_add(bpy.types.Operator):
        """Add a classic topological surface (Klein bottle, projective
        plane immersions, genus-g handlebody, solid twisted strip)"""
        bl_idname = "mesh.topological_surface_add"
        bl_label = "Topological Surface"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(name="Surface", items=PRESET_ITEMS,
                             default='KLEIN')
        res_u: IntProperty(
            name="Resolution U", default=96, min=8, max=512,
            description="Samples along u (around); for the genus "
                        "surface, implicit grid density")
        res_v: IntProperty(
            name="Resolution V", default=48, min=4, max=512,
            description="Samples along v (across / radial)")
        genus: IntProperty(
            name="Genus", default=2, min=1, max=5,
            description="Number of handles (verified for 1-5)")
        twists: IntProperty(
            name="Half-Twists", default=1, min=0, max=12,
            description="Half-twists per revolution; 1 = Mobius band")
        strip_width: FloatProperty(
            name="Strip Width", default=0.6, min=0.05, max=2.0)
        strip_thickness: FloatProperty(
            name="Strip Thickness", default=0.18, min=0.01, max=1.0)
        ridge: BoolProperty(
            name="Center Ridge", default=False,
            description="Raised ridge along the strip center line")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="Immersed surfaces only: 0 = raw surface, "
                        "> 0 = Solidify modifier of this thickness")
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            p = self.preset
            if p == 'KLEIN':
                V, F = build_klein_bottle(self.res_u, self.res_v)
                name = "Klein Bottle"
            elif p == 'KLEIN8':
                V, F = build_klein_figure8(self.res_u, self.res_v)
                name = "Klein Bottle 8"
            elif p == 'SUDANESE':
                V, F = build_sudanese_mobius(self.res_u, self.res_v)
                name = "Sudanese Mobius Band"
            elif p == 'CROSSCAP':
                V, F = build_crosscap(self.res_u, self.res_v)
                name = "Cross-Cap"
            elif p == 'ROMAN':
                V, F = build_roman(self.res_u, self.res_v)
                name = "Roman Surface"
            elif p == 'BOY':
                V, F = build_boy(self.res_u, self.res_v)
                name = "Boy Surface"
            elif p == 'GENUS':
                cell = 8.0 / max(self.res_u, 16)
                V, F = build_genus(self.genus, cell)
                name = f"Genus-{self.genus} Surface"
            else:  # TWIST_STRIP
                V, F = build_twist_strip(
                    self.twists, self.res_u, self.strip_width,
                    self.strip_thickness, self.ridge)
                name = ("Mobius Band" if self.twists == 1
                        else f"Twisted Strip ({self.twists})")
            if len(F) == 0:
                self.report({'ERROR'}, "Empty mesh")
                return {'CANCELLED'}
            # center on the origin and fit within a 2 m cube, then scale
            V = np.asarray(V, float)
            lo, hi = V.min(axis=0), V.max(axis=0)
            ext = float((hi - lo).max())
            V = (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
            obj = _new_object(context, name, V * self.scale, F,
                              smooth=self.smooth)
            if p in _IMMERSIONS and self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            p = self.preset
            if p == 'GENUS':
                lay.prop(self, 'genus')
                lay.prop(self, 'res_u')
            elif p == 'TWIST_STRIP':
                lay.prop(self, 'twists')
                lay.prop(self, 'res_u')
                lay.prop(self, 'strip_width')
                lay.prop(self, 'strip_thickness')
                lay.prop(self, 'ridge')
            else:
                lay.prop(self, 'res_u')
                lay.prop(self, 'res_v')
                lay.prop(self, 'thickness')
                if self.thickness > 0:
                    col = lay.column(align=True)
                    col.label(text="Immersed non-orientable surfaces")
                    col.label(text="thicken into their orientable")
                    col.label(text="double where they self-intersect")
                    col.label(text="(fine for viewing)")
            lay.prop(self, 'scale')
            lay.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator_menu_enum(
            "mesh.topological_surface_add", "preset",
            text="Topological Surface", icon='MESH_TORUS')

    _classes = (MESH_OT_topological_surface_add,)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


def _selftest():
    # standalone smoke tests of the numeric core
    def stats(name, V, F, chi_want, nbound_want=0):
        cnt = edge_face_counts(F)
        chi = len(V) - len(cnt) + len(F)
        nbound = sum(1 for c in cnt.values() if c == 1)
        print(f"{name:10s}: {len(V):6d} verts {len(F):6d} faces "
              f"chi = {chi:3d} (want {chi_want:3d}) "
              f"boundary edges = {nbound}")
        assert chi == chi_want and nbound == nbound_want, name

    # the Klein seams are split (2 coincident rims of nv edges
    # each), so cut open they are orientable cylinders: chi = 0
    # with 2*nv boundary edges
    V, F = build_klein_bottle(64, 32)
    stats("klein", V, F, 0, nbound_want=64)
    V, F = build_klein_figure8(64, 32)
    stats("klein8", V, F, 0, nbound_want=64)
    # split-seam Sudanese band: cut open it is a disk (chi 1); its
    # boundary is the full grid perimeter, 2*nu + 2*nv edges.
    V, F = build_sudanese_mobius(64, 32)
    stats("sudanese", V, F, 1, nbound_want=2 * 64 + 2 * 32)
    V, F = build_crosscap(64, 24)
    stats("crosscap", V, F, 1)
    V, F = build_roman(64, 24)
    stats("roman", V, F, 1)
    V, F = build_boy(64, 24)
    stats("boy", V, F, 1)
    for g in (1, 2, 3):
        V, F = build_genus(g, cell=0.125)
        stats(f"genus-{g}", V, [tuple(t) for t in F], 2 - 2 * g)
    for n in (0, 1, 2, 3):
        V, F = build_twist_strip(n, 96, ridge=(n == 1))
        cnt = edge_face_counts(F)
        ok = all(c == 2 for c in cnt.values())
        print(f"twist n={n}: {len(V)} verts, watertight = {ok}")
        assert ok
    print("standalone tests passed")
