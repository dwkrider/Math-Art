
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
# The mathematics lives in the sibling `surfaces` engine package;
# this module is the Blender layer over it.
try:
    from .surfaces.harmonics import (BOURKE_PRESETS, FORM_ITEMS, TAU,
                                         assoc_legendre,
                                         build_spherical_harmonic, math,
                                         max_abs_harmonic, np, real_sph_harm)
except ImportError:  # flat import outside the package
    from surfaces.harmonics import (BOURKE_PRESETS, FORM_ITEMS, TAU,
                                        assoc_legendre,
                                        build_spherical_harmonic, math,
                                        max_abs_harmonic, np, real_sph_harm)







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









# ==========================================================================
# The Bourke sculptural family
# ==========================================================================





# ==========================================================================
# Radial surface meshing (sphere topology, poles collapsed)
# ==========================================================================











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
