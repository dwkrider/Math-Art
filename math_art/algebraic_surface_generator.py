
# Algebraic Surface Generator for Blender
#
# Classical algebraic surfaces -- celebrated cubics, quartics,
# quintics and sextics -- built as implicit level sets f(x,y,z) = 0
# and meshed with the marching-tetrahedra extractor from the sibling
# Minimal Surface Toolkit module.
#
#   Clebsch diagonal cubic   (all 27 lines are real)
#   Cayley nodal cubic       (4 nodes -- the maximum for a cubic)
#   Kummer quartic           (16 nodes at the generic parameter)
#   Barth sextic             (65 nodes -- the maximum for a sextic)
#   Togliatti quintic        (31 nodes -- the maximum for a quintic)
#   Taubin heart, Ding-dong, Chmutov sextic, Tangle cube
#   Monkey saddle           (n-fold: z = Re((x+iy)^n), n = 3 classic)
#
# Geometry only; materials and rendering are left to Blender.
#
# References:
#   Clebsch diagonal cubic: A. Clebsch (1871). Cayley nodal cubic:
#       A. Cayley (1869). Kummer quartic: E. E. Kummer (1864).
#   Barth sextic (65 nodes): W. Barth (1996). Togliatti quintic
#       (31 nodes): E. G. Togliatti (1940). Chmutov surfaces:
#       S. V. Chmutov. Heart surface after G. Taubin (1994).
#   N-fold monkey saddles z = rho^n cos(n*phi) = Re((x+iy)^n) are the
#       graphs of the degree-n harmonic polynomials (real parts of the
#       holomorphic w^n); n = 2 is the ordinary saddle, n = 3 the
#       classic monkey saddle z = x^3 - 3xy^2. Ceramic renditions of
#       these saddle sheets recur in Robert Fathauer's mathematical
#       ceramics (his n-fold saddle forms).

bl_info = {
    "name": "Algebraic Surface Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Algebraic Surface",
    "description": "Classical algebraic surfaces (Clebsch, Cayley, "
                   "Kummer, Barth, Togliatti, ...) as implicit level "
                   "sets meshed by marching tetrahedra",
    "category": "Add Mesh",
}

import math
import numpy as np

_PHI = (1.0 + math.sqrt(5.0)) / 2.0          # golden ratio
_SQRT2 = math.sqrt(2.0)


def _toolkit():
    """The sibling Minimal Surface Toolkit supplies marching_tets.
    Relative import when installed as part of the Math Art package,
    absolute when this file runs standalone next to the toolkit."""
    try:
        from . import minimal_surface_toolkit as mst
    except ImportError:
        import minimal_surface_toolkit as mst
    return mst


# ==========================================================================
# Implicit fields
# ==========================================================================
# Each field takes numpy sample grids x, y, z plus the Kummer
# parameter mu (ignored by every preset except the Kummer quartic)
# and returns f on the grid; the surface is the zero level set.

def _f_clebsch(x, y, z, mu):
    # 81(x^3+y^3+z^3) - 189(sum of x^2 y terms) + 54xyz
    #   + 126(xy+xz+yz) - 9(x^2+y^2+z^2) - 9(x+y+z) + 1 = 0
    # Future option: overlay the famous 27 real lines of this cubic
    # as curve objects (not implemented).
    x2, y2, z2 = x * x, y * y, z * z
    return (81.0 * (x2 * x + y2 * y + z2 * z)
            - 189.0 * (x2 * y + x2 * z + y2 * x + y2 * z
                       + z2 * x + z2 * y)
            + 54.0 * x * y * z
            + 126.0 * (x * y + x * z + y * z)
            - 9.0 * (x2 + y2 + z2)
            - 9.0 * (x + y + z) + 1.0)


def _f_cayley(x, y, z, mu):
    # 4(x^2+y^2+z^2) + 16xyz - 1 = 0 -- four conical nodes, the
    # maximum possible on a cubic surface
    return 4.0 * (x * x + y * y + z * z) + 16.0 * x * y * z - 1.0


def _f_kummer(x, y, z, mu):
    # (x^2+y^2+z^2 - mu^2)^2 - lambda p q r s = 0 with the four
    # tetrahedral tangent planes p, q, r, s; 16 nodes for generic mu
    mu2 = mu * mu
    lam = (3.0 * mu2 - 1.0) / (3.0 - mu2)
    p = 1.0 - z - _SQRT2 * x
    q = 1.0 - z + _SQRT2 * x
    r = 1.0 + z + _SQRT2 * y
    s = 1.0 + z - _SQRT2 * y
    core = x * x + y * y + z * z - mu2
    return core * core - lam * p * q * r * s


def _f_barth(x, y, z, mu):
    # 4(phi^2 x^2 - y^2)(phi^2 y^2 - z^2)(phi^2 z^2 - x^2)
    #   - (1+2 phi)(x^2+y^2+z^2-1)^2 = 0, phi the golden ratio
    p2 = _PHI * _PHI
    x2, y2, z2 = x * x, y * y, z * z
    w = x2 + y2 + z2 - 1.0
    return (4.0 * (p2 * x2 - y2) * (p2 * y2 - z2) * (p2 * z2 - x2)
            - (1.0 + 2.0 * _PHI) * w * w)


def _f_togliatti(x, y, z, mu):
    # 64(x-1)(x^4 - 4x^3 - 10x^2 y^2 - 4x^2 + 16x - 20x y^2
    #   + 5y^4 + 16 - 20y^2)
    #   - 5 a (2z - a)(4(x^2+y^2-z^2) + (1+3 sqrt5))^2 = 0,
    # a = sqrt(5 - sqrt5)
    a = math.sqrt(5.0 - math.sqrt(5.0))
    x2, y2 = x * x, y * y
    quart = (x2 * x2 - 4.0 * x2 * x - 10.0 * x2 * y2 - 4.0 * x2
             + 16.0 * x - 20.0 * x * y2 + 5.0 * y2 * y2
             + 16.0 - 20.0 * y2)
    q = 4.0 * (x2 + y2 - z * z) + (1.0 + 3.0 * math.sqrt(5.0))
    return (64.0 * (x - 1.0) * quart
            - 5.0 * a * (2.0 * z - a) * q * q)


def _f_heart(x, y, z, mu):
    # Taubin's heart (z up, lobes on top):
    # (x^2 + 9/4 y^2 + z^2 - 1)^3 - x^2 z^3 - 9/80 y^2 z^3 = 0
    z3 = z * z * z
    w = x * x + 2.25 * y * y + z * z - 1.0
    return w * w * w - x * x * z3 - 0.1125 * y * y * z3


def _f_dingdong(x, y, z, mu):
    # x^2 + y^2 - (1 - z) z^2 = 0 -- a droplet sitting on a cone
    return x * x + y * y - (1.0 - z) * z * z


def _f_chmutov(x, y, z, mu):
    # T6(x) + T6(y) + T6(z) = 0 with the degree-6 Chebyshev
    # polynomial T6(t) = 32t^6 - 48t^4 + 18t^2 - 1
    def t6(t):
        t2 = t * t
        return ((32.0 * t2 - 48.0) * t2 + 18.0) * t2 - 1.0
    return t6(x) + t6(y) + t6(z)


def _f_tangle(x, y, z, mu):
    # tangle cube: x^4 - 5x^2 + y^4 - 5y^2 + z^4 - 5z^2 + 11.8 = 0
    x2, y2, z2 = x * x, y * y, z * z
    return (x2 * x2 - 5.0 * x2 + y2 * y2 - 5.0 * y2
            + z2 * z2 - 5.0 * z2 + 11.8)


def _f_monkey(x, y, z, mu, n=3):
    # n-fold monkey saddle: z = Re((x+iy)^n) = rho^n cos(n*phi), the
    # graph of the degree-n harmonic polynomial (real part of the
    # holomorphic w^n).  n = 2 is the ordinary saddle z = x^2 - y^2,
    # n = 3 the classic monkey saddle z = x^3 - 3xy^2 (two legs and a
    # tail), n >= 4 the higher-fold saddles -- forms Robert Fathauer
    # renders as ceramic saddle sheets.  Scaling: inside the unit clip
    # ball rho <= 1 the height obeys |Re w^n| <= rho^n <= 1, so the
    # unit-coefficient polynomial already sits in frame; the spherical
    # clip trims the sheet to a rim that waves up and down n times.
    return z - ((x + 1j * y) ** int(n)).real


# Each preset stores its own clip region framing the interesting part
# of the (usually unbounded) surface: 'BALL' with radius r, or 'BOX'
# with half-extent r.
PRESETS = {
    'CLEBSCH': ("Clebsch Diagonal Cubic", _f_clebsch, 'BALL', 3.0),
    'CAYLEY': ("Cayley Nodal Cubic", _f_cayley, 'BALL', 1.4),
    'KUMMER': ("Kummer Quartic", _f_kummer, 'BALL', 2.2),
    'BARTH': ("Barth Sextic", _f_barth, 'BALL', 2.0),
    'TOGLIATTI': ("Togliatti Quintic", _f_togliatti, 'BALL', 5.0),
    'HEART': ("Taubin Heart", _f_heart, 'BOX', 1.5),
    'DINGDONG': ("Ding-dong", _f_dingdong, 'BALL', 1.5),
    'CHMUTOV': ("Chmutov Sextic", _f_chmutov, 'BOX', 1.1),
    'TANGLE': ("Tangle Cube", _f_tangle, 'BOX', 2.4),
    'MONKEY': ("Monkey Saddle (n-fold)", _f_monkey, 'BALL', 1.0),
}


def build_algebraic(kind, res, mu=1.3, clip=0.0, scale=1.0, fold=3):
    """Mesh the zero level set of a preset. Returns (verts, tris).
    marching_tets simply leaves the level set open where it crosses
    the sample box, which for the BOX presets is exactly the wanted
    clip. BALL presets sample the bounding cube of the ball and then
    cull triangles whose centroid falls outside it -- an open, even
    rim (masking outside samples to a large positive value instead
    stitches jagged stair-step caps onto the boundary). `fold` is the
    monkey-saddle fold count n (MONKEY preset only)."""
    label, fn, shape, clip_default = PRESETS[kind]
    r = clip if clip > 0.0 else clip_default
    if kind == 'MONKEY':
        n = max(2, int(round(fold)))
        field = lambda X, Y, Z: fn(X, Y, Z, mu, n)
    else:
        field = lambda X, Y, Z: fn(X, Y, Z, mu)
    mst = _toolkit()
    verts, tris = mst.marching_tets(
        field, (-r, -r, -r), (r, r, r), (res, res, res))
    if shape == 'BALL' and len(tris):
        cen = verts[tris].mean(axis=1)
        keep = np.einsum('ij,ij->i', cen, cen) <= r * r
        tris = tris[keep]
        used = np.unique(tris)               # drop orphaned verts
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        verts = verts[used]
        tris = remap[tris]
    # center on the origin and fit within a 2 m cube, then apply scale
    if len(verts):
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        ext = float((hi - lo).max())
        verts = (verts - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9
                                             else 1.0)
    return verts * scale, tris


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


if _IN_BLENDER:

    def _new_object(context, name, verts, faces, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
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

    class MESH_OT_algebraic_surface_add(bpy.types.Operator):
        """Add a classical algebraic surface (implicit level set
        meshed by marching tetrahedra)"""
        bl_idname = "mesh.algebraic_surface_add"
        bl_label = "Algebraic Surface"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset",
            items=[(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='CLEBSCH')
        resolution: IntProperty(
            name="Resolution", default=80, min=16, max=256,
            description="Sample grid resolution per axis (algebraic "
                        "surfaces need more than TPMS)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        mu: FloatProperty(
            name="Kummer Mu", default=1.3, min=1.05, max=2.0,
            description="Kummer quartic parameter (node sharpness); "
                        "used by the Kummer preset only")
        fold: IntProperty(
            name="Fold n", default=3, min=2, max=8,
            description="Saddle fold count: 2 = ordinary saddle, "
                        "3 = monkey saddle, higher = n-fold saddles; "
                        "Monkey Saddle preset only")
        clip: FloatProperty(
            name="Clip Override", default=0.0, min=0.0, max=20.0,
            description="Clip ball radius / box half-extent; "
                        "0 uses the preset default")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this "
                        "thickness")
        smooth: BoolProperty(
            name="Smooth Shading", default=True)

        def execute(self, context):
            label = PRESETS[self.preset][0]
            verts, tris = build_algebraic(
                self.preset, self.resolution, mu=self.mu,
                clip=self.clip, scale=self.scale, fold=self.fold)
            if len(tris) == 0:
                self.report({'ERROR'}, "Empty level set")
                return {'CANCELLED'}
            obj = _new_object(context, label, verts, tris,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            me = obj.data
            self.report({'INFO'},
                        f"{label}: {len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'resolution')
            if self.preset == 'KUMMER':
                lay.prop(self, 'mu')
            if self.preset == 'MONKEY':
                lay.prop(self, 'fold')
            for k in ('clip', 'scale', 'thickness', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.algebraic_surface_add",
                             icon='SURFACE_NSPHERE')

    _classes = (MESH_OT_algebraic_surface_add,)

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
    # standalone smoke test of the numeric core (requires the
    # Minimal Surface Toolkit importable as a sibling)
    for kind in PRESETS:
        V, T = build_algebraic(kind, 40)
        print(f"{kind:10s}: {len(V):6d} verts {len(T):6d} tris")
