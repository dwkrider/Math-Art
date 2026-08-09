
# Spherical Harmonic Generator for Blender
#
# The spherical harmonics Y_l^m are the eigenfunctions of the
# Laplace-Beltrami operator on the sphere -- the angular part of every
# separable solution of Laplace's equation, and so the shape language
# of gravitational and electrostatic multipoles, vibrating shells and
# atomic orbitals alike.  This module renders them as radial surfaces
# r = f(theta, phi) over the sphere of directions, in four forms:
#
#   OFFSET  r = r0 + a Y_l^m -- a gently deformed sphere.  Always
#           star-shaped (hence embedded and printable) while
#           a < r0 / max|Y|; the safe default.
#   ABS     r = |Y_l^m| -- the classic lobed "balloon" picture.  The
#           nodal circles pinch to points, so the surface touches
#           itself there; it is an immersion, not an embedding.
#   SIGNED  as ABS, with the lobes split by the sign of Y into two
#           material slots (or into separate loose parts).  This is
#           the picture chemists draw for orbital angular parts; the
#           sibling Atomic & Molecular Orbital module supplies the
#           radial part that turns it into a real wavefunction.
#   BOURKE  the eight-integer sculptural family
#           r = sin(m0 phi)^m1 + cos(m2 phi)^m3
#             + sin(m4 theta)^m5 + cos(m6 theta)^m7,
#           which is not a spherical harmonic at all but a popular
#           trigonometric relative of one -- a rich source of lobed,
#           flower-like solids.
#
# Real spherical harmonics are used throughout (the real and imaginary
# parts of the complex Y_l^m, which is what one actually wants to
# draw), with the Condon-Shortley phase included.  The associated
# Legendre functions come from the standard three-term recurrence:
# Blender ships no scipy, so every special function here is hand-rolled
# on numpy arrays.
#
# Geometry only; materials and rendering are left to Blender.
#
# References:
# - Spherical harmonics: P. S. Laplace, "Theorie des attractions des
#   spheroides et de la figure des planetes", Memoires de l'Academie
#   royale des Sciences, 1785; A.-M. Legendre, "Recherches sur
#   l'attraction des spheroides homogenes", Memoires de Mathematique
#   et de Physique, 1785.
# - Real form, normalisation and the Condon-Shortley phase:
#   E. U. Condon and G. H. Shortley, "The Theory of Atomic Spectra",
#   Cambridge University Press, 1935.
# - The associated Legendre recurrences used here: M. Abramowitz and
#   I. A. Stegun, "Handbook of Mathematical Functions", Dover, 1965,
#   chapter 8.
# - The eight-parameter sculptural family
#     r = sin(m0 phi)^m1 + cos(m2 phi)^m3 + sin(m4 theta)^m5
#         + cos(m6 theta)^m7
#   is Paul Bourke's "Spherical Harmonics" form (February 1990),
#   http://paulbourke.net/geometry/sphericalh/ .  The parameter sets
#   offered as presets below are project-chosen, not his.

bl_info = {
    "name": "Spherical Harmonic Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Spherical Harmonic",
    "description": "Real spherical harmonics Y_l^m as radial surfaces "
                   "(offset, absolute and sign-split forms) plus the "
                   "eight-integer Bourke harmonic family",
    "category": "Add Mesh",
}

import math

import numpy as np

TAU = 2.0 * math.pi


# ==========================================================================
# Associated Legendre functions and real spherical harmonics
# ==========================================================================
# No scipy in Blender, so P_l^m comes from the classical recurrences
#
#   P_m^m(x)   = (-1)^m (2m-1)!! (1-x^2)^(m/2)        [Condon-Shortley]
#   P_(m+1)^m  = x (2m+1) P_m^m
#   (l-m) P_l^m = x (2l-1) P_(l-1)^m - (l+m-1) P_(l-2)^m
#
# evaluated on whole numpy arrays at once.

def assoc_legendre(l, m, x):
    """Associated Legendre function P_l^m(x) on an array x in [-1, 1],
    Condon-Shortley phase included.  Requires 0 <= m <= l."""
    l, m = int(l), int(abs(m))
    if m > l:
        raise ValueError(f"assoc_legendre needs |m| <= l, got l={l}, m={m}")
    x = np.asarray(x, dtype=float)
    pmm = np.ones_like(x)
    if m > 0:
        # (1-x^2)^(m/2), clamped: rounding can push |x| a hair over 1
        somx2 = np.sqrt(np.maximum(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(m):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = pmmp1
    for ll in range(m + 2, l + 1):
        pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


def sph_harm_norm(l, m):
    """The real-harmonic normalisation sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!),
    computed through lgamma so large l cannot overflow."""
    l, m = int(l), int(abs(m))
    return math.exp(0.5 * (math.log(2.0 * l + 1.0) - math.log(4.0 * math.pi)
                           + math.lgamma(l - m + 1.0)
                           - math.lgamma(l + m + 1.0)))


def real_sph_harm(l, m, theta, phi):
    """Real spherical harmonic Y_l^m evaluated on arrays of the polar
    angle theta in [0, pi] (colatitude, 0 at +z) and the azimuth phi.

    Sign convention: m > 0 takes the cosine (cos m phi) partner, m < 0
    the sine partner, and both carry the sqrt(2) that keeps the real
    pair orthonormal.  This is the standard real basis used for atomic
    orbitals: (l, m) = (1, 0) is p_z, (1, 1) is p_x, (1, -1) is p_y."""
    l, m = int(l), int(m)
    am = abs(m)
    if am > l:
        raise ValueError(f"real_sph_harm needs |m| <= l, got l={l}, m={m}")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    p = assoc_legendre(l, am, np.cos(theta))
    n = sph_harm_norm(l, am)
    if m == 0:
        return n * p
    if m > 0:
        return math.sqrt(2.0) * n * p * np.cos(am * phi)
    return math.sqrt(2.0) * n * p * np.sin(am * phi)


def max_abs_harmonic(l, m, samples=512):
    """max |Y_l^m| over the sphere -- the scale the OFFSET form needs in
    order to guarantee a positive (hence star-shaped) radius."""
    th = np.linspace(0.0, math.pi, samples)
    p = assoc_legendre(l, abs(m), np.cos(th))
    n = sph_harm_norm(l, abs(m))
    amp = n * float(np.max(np.abs(p)))
    return amp if m == 0 else math.sqrt(2.0) * amp


# ==========================================================================
# The Bourke sculptural family
# ==========================================================================

def bourke_radius(mm, theta, phi):
    """r = sin(m0 phi)^m1 + cos(m2 phi)^m3 + sin(m4 theta)^m5
           + cos(m6 theta)^m7
    with phi the polar angle in [0, pi] and theta the azimuth in
    [0, 2 pi], following Bourke's own convention.

    The exponents MUST be Python ints: numpy raises a negative float
    base to a float power as NaN, which is the classic way to get an
    empty mesh out of this family."""
    m0, m1, m2, m3, m4, m5, m6, m7 = (int(v) for v in mm)
    return (np.sin(m0 * phi) ** m1 + np.cos(m2 * phi) ** m3
            + np.sin(m4 * theta) ** m5 + np.cos(m6 * theta) ** m7)


# Project-chosen parameter tuples (Bourke publishes renders, not named
# sets).  Each is (label, (m0..m7)).
BOURKE_PRESETS = [
    ('B1', "Bourke 4-1-4-1", (4, 1, 4, 1, 4, 1, 4, 1)),
    ('B2', "Bourke 2-1-2-1", (2, 1, 2, 1, 2, 1, 2, 1)),
    ('B3', "Bourke 1-2-2-2", (1, 2, 2, 2, 4, 2, 3, 2)),
    ('B4', "Bourke 3-2-2-3", (3, 2, 2, 3, 3, 2, 2, 3)),
    ('B5', "Bourke 5-1-3-1", (5, 1, 3, 1, 5, 1, 3, 1)),
    ('B6', "Bourke 2-3-4-1", (2, 3, 4, 1, 2, 3, 4, 1)),
]


# ==========================================================================
# Radial surface meshing (sphere topology, poles collapsed)
# ==========================================================================

def build_radial_surface(rfun, nu=128, nv=256):
    """Mesh r = rfun(theta_polar, phi_azimuth) as a closed sphere-topology
    surface.  The two poles collapse to single vertices and the azimuth
    seam is glued by index, so the result has chi = 2 and no boundary.

    Returns (verts (n,3), faces list, face_param (m,2)) where face_param
    holds the (theta, phi) of each face centre -- the SIGNED form uses it
    to look up the sign of Y without re-deriving it from geometry."""
    nu, nv = max(4, int(nu)), max(6, int(nv))
    th = math.pi * np.arange(nu + 1) / nu            # 0 .. pi
    ph = TAU * np.arange(nv) / nv                    # 0 .. 2pi (wrapped)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    R = np.asarray(rfun(TH, PH), dtype=float)
    if R.shape != TH.shape:
        R = np.broadcast_to(R, TH.shape)
    X = R * np.sin(TH) * np.cos(PH)
    Y = R * np.sin(TH) * np.sin(PH)
    Z = R * np.cos(TH)

    # poles: every column of row 0 (and row nu) is the same point, so
    # collapse each to one vertex -- welding by index, never by
    # coordinate proximity
    verts = [(0.0, 0.0, float(np.mean(Z[0])))]
    idx = np.zeros((nu + 1, nv), dtype=np.int64)
    idx[0, :] = 0
    for i in range(1, nu):
        base = len(verts)
        for j in range(nv):
            verts.append((float(X[i, j]), float(Y[i, j]), float(Z[i, j])))
        idx[i, :] = base + np.arange(nv)
    south = len(verts)
    verts.append((0.0, 0.0, float(np.mean(Z[nu]))))
    idx[nu, :] = south

    faces, fparam = [], []

    def mid(i0, i1, j0, j1):
        return (0.5 * (th[i0] + th[i1]),
                0.5 * (ph[j0] + ph[j0] + TAU / nv))

    for j in range(nv):
        j2 = (j + 1) % nv
        faces.append((0, idx[1, j2], idx[1, j]))
        fparam.append(mid(0, 1, j, j2))
    for i in range(1, nu - 1):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append((idx[i, j], idx[i, j2],
                          idx[i + 1, j2], idx[i + 1, j]))
            fparam.append(mid(i, i + 1, j, j2))
    for j in range(nv):
        j2 = (j + 1) % nv
        faces.append((south, idx[nu - 1, j], idx[nu - 1, j2]))
        fparam.append(mid(nu - 1, nu, j, j2))

    return np.asarray(verts, dtype=float), faces, np.asarray(fparam)


def _drop_faces(verts, faces, extra, keep):
    """Keep the faces flagged by `keep`, dropping orphaned vertices.
    `extra` is a per-face array carried along."""
    faces = [f for f, k in zip(faces, keep) if k]
    extra = [e for e, k in zip(extra, keep) if k]
    used = sorted({i for f in faces for i in f})
    remap = {o: n for n, o in enumerate(used)}
    return (verts[used], [tuple(remap[i] for i in f) for f in faces],
            np.asarray(extra))


def center_fit(verts, scale=1.0):
    """Centre on the bounding-box midpoint and fit the largest extent to
    a 2 m cube, then apply `scale` (the project-wide convention)."""
    if not len(verts):
        return verts
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ext = float((hi - lo).max())
    out = (verts - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
    return out * scale


FORM_ITEMS = [
    ('OFFSET', "Offset Sphere", "r = r0 + a Y_l^m: a smoothly deformed "
                                "sphere, always embedded"),
    ('ABS', "Absolute (lobes)", "r = |Y_l^m|: the classic lobed balloon "
                                "(pinched at the nodal circles)"),
    ('SIGNED', "Signed lobes", "r = |Y_l^m| with the lobes separated by "
                               "the sign of Y"),
    ('BOURKE', "Bourke Family", "The eight-integer trigonometric "
                                "harmonic family"),
]


def build_spherical_harmonic(form='OFFSET', l=3, m=2, nu=128, nv=256,
                             r0=1.0, amp=0.6, eps=0.02, mm=None,
                             abs_radius=False, split_lobes=False,
                             scale=1.0):
    """Build one spherical-harmonic surface.  Returns
    (verts, faces, face_sign) with face_sign +1/-1 per face for the
    SIGNED form (all +1 otherwise)."""
    l = max(0, int(l))
    m = int(np.clip(int(m), -l, l))

    if form == 'BOURKE':
        mm = tuple(mm) if mm else BOURKE_PRESETS[0][2]

        def rfun(theta, phi):
            # Bourke's phi is the POLAR angle and his theta the azimuth;
            # our grid hands them over in (polar, azimuth) order, so his
            # (theta, phi) is our (phi, theta).
            r = bourke_radius(mm, phi, theta)
            return np.abs(r) if abs_radius else r
    elif form == 'OFFSET':
        def rfun(theta, phi):
            return r0 + amp * real_sph_harm(l, m, theta, phi)
    else:
        def rfun(theta, phi):
            return np.abs(real_sph_harm(l, m, theta, phi)) + eps

    verts, faces, fparam = build_radial_surface(rfun, nu, nv)

    sign = np.ones(len(faces), dtype=int)
    if form == 'SIGNED':
        y = real_sph_harm(l, m, fparam[:, 0], fparam[:, 1])
        sign = np.where(y >= 0.0, 1, -1)
        if split_lobes:
            # drop the band of faces straddling a nodal line so the
            # lobes come apart into separate loose parts
            keep = np.abs(y) > 0.02 * max(max_abs_harmonic(l, m), 1e-9)
            verts, faces, fparam = _drop_faces(verts, faces, fparam, keep)
            y = real_sph_harm(l, m, fparam[:, 0], fparam[:, 1])
            sign = np.where(y >= 0.0, 1, -1)

    return center_fit(verts, scale), faces, sign


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

    _LOBE_COLORS = {1: (0.85, 0.25, 0.20), -1: (0.20, 0.40, 0.85)}

    def _lobe_material(s):
        name = "Harmonic Lobe +" if s > 0 else "Harmonic Lobe -"
        mat = bpy.data.materials.get(name)
        if mat is None:
            rgb = _LOBE_COLORS[s]
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.45
        return mat

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

    class MESH_OT_spherical_harmonic_add(bpy.types.Operator):
        """Add a spherical harmonic Y_l^m as a radial surface, or a
        member of Bourke's eight-integer harmonic family"""
        bl_idname = "mesh.spherical_harmonic_add"
        bl_label = "Spherical Harmonic"
        bl_options = {'REGISTER', 'UNDO'}

        form: EnumProperty(name="Form", items=FORM_ITEMS,
                           default='OFFSET')
        degree: IntProperty(
            name="Degree l", default=3, min=0, max=12,
            description="Degree l of the harmonic: the surface has l "
                        "nodal circles in total")
        order: IntProperty(
            name="Order m", default=2, min=-12, max=12,
            description="Order m of the harmonic; clamped to |m| <= l")
        r0: FloatProperty(
            name="Base Radius", default=1.0, min=0.05, max=10.0,
            description="Offset form: the undeformed sphere radius")
        amp: FloatProperty(
            name="Amplitude", default=0.6, min=-5.0, max=5.0,
            description="Offset form: how strongly Y_l^m deforms the "
                        "sphere")
        eps: FloatProperty(
            name="Nodal Gap", default=0.02, min=0.0, max=0.5,
            description="Absolute/signed forms: radius added at the "
                        "nodal circles so they do not pinch to a point")
        split_lobes: BoolProperty(
            name="Split Lobes", default=False,
            description="Signed form: separate the lobes into loose "
                        "parts instead of joining them at the nodes")
        bourke_preset: EnumProperty(
            name="Bourke Set",
            items=[(k, lab, lab) for k, lab, _ in BOURKE_PRESETS]
                  + [('CUSTOM', "Custom", "Use the m0..m7 fields")],
            default='B1')
        m0: IntProperty(name="m0", default=4, min=0, max=8)
        m1: IntProperty(name="m1", default=1, min=0, max=8)
        m2: IntProperty(name="m2", default=4, min=0, max=8)
        m3: IntProperty(name="m3", default=1, min=0, max=8)
        m4: IntProperty(name="m4", default=4, min=0, max=8)
        m5: IntProperty(name="m5", default=1, min=0, max=8)
        m6: IntProperty(name="m6", default=4, min=0, max=8)
        m7: IntProperty(name="m7", default=1, min=0, max=8)
        abs_radius: BoolProperty(
            name="Absolute Radius", default=False,
            description="Bourke form: use |r|, so the surface cannot "
                        "fold through the origin")
        res_u: IntProperty(
            name="Resolution (polar)", default=128, min=8, max=1024)
        res_v: IntProperty(
            name="Resolution (azimuth)", default=256, min=8, max=1024)
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this "
                        "thickness")
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            l, m = self.degree, self.order
            if abs(m) > l:
                self.report({'WARNING'},
                            f"|m| must be <= l; clamping m to "
                            f"{int(np.clip(m, -l, l))}")
            if self.bourke_preset == 'CUSTOM':
                mm = (self.m0, self.m1, self.m2, self.m3,
                      self.m4, self.m5, self.m6, self.m7)
            else:
                mm = dict((k, v) for k, _, v in BOURKE_PRESETS)[
                    self.bourke_preset]
            verts, faces, sign = build_spherical_harmonic(
                form=self.form, l=l, m=m, nu=self.res_u, nv=self.res_v,
                r0=self.r0, amp=self.amp, eps=self.eps, mm=mm,
                abs_radius=self.abs_radius,
                split_lobes=self.split_lobes, scale=self.scale)
            if not len(faces):
                self.report({'ERROR'}, "Empty surface")
                return {'CANCELLED'}
            if self.form == 'BOURKE':
                label = f"Bourke Harmonic {'-'.join(str(v) for v in mm)}"
            else:
                label = f"Y({l},{int(np.clip(m, -l, l))})"
            obj = _new_object(context, label, verts, faces,
                              smooth=self.smooth)
            me = obj.data
            if self.form == 'SIGNED' and len(me.polygons) == len(faces):
                me.materials.append(_lobe_material(1))
                me.materials.append(_lobe_material(-1))
                me.polygons.foreach_set(
                    'material_index',
                    [0 if s > 0 else 1 for s in sign])
                me.update()
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            self.report({'INFO'},
                        f"{label}: {len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'form')
            if self.form == 'BOURKE':
                lay.prop(self, 'bourke_preset')
                if self.bourke_preset == 'CUSTOM':
                    row = lay.row(align=True)
                    for k in ('m0', 'm1', 'm2', 'm3'):
                        row.prop(self, k)
                    row = lay.row(align=True)
                    for k in ('m4', 'm5', 'm6', 'm7'):
                        row.prop(self, k)
                lay.prop(self, 'abs_radius')
            else:
                lay.prop(self, 'degree')
                lay.prop(self, 'order')
                if self.form == 'OFFSET':
                    lay.prop(self, 'r0')
                    lay.prop(self, 'amp')
                else:
                    lay.prop(self, 'eps')
                if self.form == 'SIGNED':
                    lay.prop(self, 'split_lobes')
            for k in ('res_u', 'res_v', 'scale', 'thickness', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.spherical_harmonic_add",
                             icon='SURFACE_NSPHERE')

    _classes = (MESH_OT_spherical_harmonic_add,)

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


# ==========================================================================
# Standalone numeric self-test
# ==========================================================================

def _mesh_stats(verts, faces):
    """(nverts, nedges, nfaces, chi, nboundary) for a face list."""
    cnt = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    nb = sum(1 for v in cnt.values() if v == 1)
    return (len(verts), len(cnt), len(faces),
            len(verts) - len(cnt) + len(faces), nb)


def _selftest():
    # ---- 1. orthonormality of the real harmonics on the sphere ------
    # Gauss-Legendre in cos(theta) x uniform in phi integrates these
    # trigonometric polynomials essentially exactly.
    nodes, weights = np.polynomial.legendre.leggauss(96)
    theta = np.arccos(nodes)
    nphi = 256
    phi = TAU * np.arange(nphi) / nphi
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    W = weights[:, None] * np.full((1, nphi), TAU / nphi)
    basis = [(l, m) for l in range(5) for m in range(-l, l + 1)]
    fields = {lm: real_sph_harm(lm[0], lm[1], TH, PH) for lm in basis}
    worst = 0.0
    for i, a in enumerate(basis):
        for b in basis[i:]:
            val = float(np.sum(fields[a] * fields[b] * W))
            want = 1.0 if a == b else 0.0
            worst = max(worst, abs(val - want))
    if worst > 1e-6:
        raise AssertionError(
            f"real spherical harmonics not orthonormal: worst "
            f"deviation {worst:.3e}")
    print(f"orthonormality (l <= 4): worst deviation {worst:.2e}")

    # ---- 2. Y_0^0 is the constant 1 / (2 sqrt(pi)) -------------------
    y00 = real_sph_harm(0, 0, TH, PH)
    if abs(float(np.max(np.abs(y00 - 0.5 / math.sqrt(math.pi))))) > 1e-12:
        raise AssertionError("Y(0,0) is not the constant 1/(2 sqrt pi)")

    # ---- 3. nodal counts --------------------------------------------
    # Y_l^m has l - |m| nodal circles of latitude and 2|m| nodal
    # meridians; a wrong Legendre index shows up here first.
    tt = np.linspace(1e-3, math.pi - 1e-3, 4001)
    for l in range(6):
        for m in range(-l, l + 1):
            am = abs(m)
            merid = real_sph_harm(l, m, tt, np.zeros_like(tt))
            if m < 0:
                # the sin(m phi) partner vanishes identically at phi = 0
                merid = real_sph_harm(l, m, tt,
                                      np.full_like(tt, math.pi / (2 * am)))
            s = np.sign(merid)
            nz = int(np.sum(s[1:] * s[:-1] < 0))
            if nz != l - am:
                raise AssertionError(
                    f"Y({l},{m}) has {nz} latitude nodes, expected "
                    f"{l - am}")
            if am:
                # sample the azimuth where |P_l^m| is largest
                p = np.abs(assoc_legendre(l, am, np.cos(tt)))
                th_star = float(tt[int(np.argmax(p))])
                # half-step offsets so no sample lands exactly on a
                # node (sign(0) is 0 and would swallow a crossing)
                pp = TAU * (np.arange(4000) + 0.5) / 4000
                eq = real_sph_harm(l, m, np.full_like(pp, th_star), pp)
                se = np.sign(eq)
                nz2 = int(np.sum(se[1:] * se[:-1] < 0))
                nz2 += 1 if se[0] * se[-1] < 0 else 0
                if nz2 != 2 * am:
                    raise AssertionError(
                        f"Y({l},{m}) has {nz2} meridian nodes, "
                        f"expected {2 * am}")
    print("nodal counts (l <= 5): latitude and meridian nodes correct")

    # ---- 4. closed forms --------------------------------------------
    # Y(1,0) = sqrt(3/4pi) cos theta, Y(2,0) = sqrt(5/16pi)(3cos^2-1)
    c = np.cos(TH)
    ref10 = math.sqrt(3.0 / (4.0 * math.pi)) * c
    ref20 = math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * c * c - 1.0)
    for lm, ref in (((1, 0), ref10), ((2, 0), ref20)):
        err = float(np.max(np.abs(fields[lm] - ref)))
        if err > 1e-12:
            raise AssertionError(f"Y{lm} differs from its closed form "
                                 f"by {err:.3e}")

    # ---- 5. meshes ---------------------------------------------------
    for form, l, m in (('OFFSET', 3, 2), ('ABS', 3, 2), ('SIGNED', 4, -3),
                       ('OFFSET', 0, 0), ('ABS', 5, 5)):
        V, F, S = build_spherical_harmonic(form=form, l=l, m=m,
                                           nu=48, nv=96)
        nv_, ne, nf, chi, nb = _mesh_stats(V, F)
        if chi != 2 or nb != 0:
            raise AssertionError(
                f"{form} Y({l},{m}): expected a closed sphere "
                f"(chi 2, no boundary), got chi {chi}, {nb} boundary "
                f"edges")
        if not np.all(np.isfinite(V)):
            raise AssertionError(f"{form} Y({l},{m}) produced "
                                 f"non-finite vertices")
        ext = float((V.max(axis=0) - V.min(axis=0)).max())
        if abs(ext - 2.0) > 1e-6:
            raise AssertionError(f"{form} Y({l},{m}) is {ext:.4f} across, "
                                 f"expected a 2 m fit")
        print(f"{form:7s} Y({l:2d},{m:2d}): {nv_:6d} verts {nf:6d} faces "
              f"chi {chi}")

    # star-shapedness of the OFFSET default: amp below r0/max|Y| must
    # keep the radius positive everywhere
    for l in range(6):
        for m in range(-l, l + 1):
            mx = max_abs_harmonic(l, m)
            r = 1.0 + (0.95 / mx) * real_sph_harm(l, m, TH, PH)
            if float(np.min(r)) <= 0.0:
                raise AssertionError(
                    f"offset Y({l},{m}) radius went non-positive")

    # ---- 6. sign split ----------------------------------------------
    V, F, S = build_spherical_harmonic(form='SIGNED', l=1, m=0,
                                       nu=48, nv=96)
    pos, neg = int(np.sum(S > 0)), int(np.sum(S < 0))
    if pos == 0 or neg == 0:
        raise AssertionError(f"signed p_z should have both lobes, got "
                             f"{pos} positive / {neg} negative faces")
    if abs(pos - neg) > 0.02 * len(S):
        raise AssertionError("signed p_z lobes are not balanced")
    print(f"signed Y(1,0): {pos} positive / {neg} negative faces")

    # ---- 7. Bourke family: finite, no NaN from integer powers --------
    for key, label, mm in BOURKE_PRESETS:
        V, F, S = build_spherical_harmonic(form='BOURKE', mm=mm,
                                           nu=48, nv=96)
        if not np.all(np.isfinite(V)):
            raise AssertionError(f"{label} produced non-finite vertices "
                                 f"(a float exponent leaked in)")
        nv_, ne, nf, chi, nb = _mesh_stats(V, F)
        if chi != 2 or nb != 0:
            raise AssertionError(f"{label}: chi {chi}, {nb} boundary "
                                 f"edges; expected a closed surface")
        print(f"{label:16s}: {nv_:6d} verts {nf:6d} faces")

    print("RESULT: OK")
