
# Harmonic Knots generator for Blender: knots parametrized by pure
# harmonics -- Lissajous, Fourier, Chebyshev and billiard curves --
# with the same output styles as the Torus Knot generator (bezier /
# poly / NURBS curve or swept tube mesh).
#
# LISSAJOUS: (cos(nx t + px), cos(ny t + py), cos(nz t + pz)) with
# nx, ny, nz pairwise coprime.  FOURIER generalizes each coordinate
# to a finite sum of cosines; with one extra z-harmonic these are
# the Fourier-(1,1,2) knots, and Hoste proved every torus knot is
# one (the built-in preset realizes the (2,3) trefoil this way; its
# phases were fitted numerically, the knot type is verified by the
# determinant check in the self-test below).  CHEBYSHEV knots are
# polynomial arcs (T_a(s), T_b(s), T_c(s + phi)) in the Chebyshev
# polynomials of the first kind, closed here by a one-bridge arc
# routed above the diagram (every knot arises this way; note
# T_n(cos t) = cos(n t), so on [-1, 1] these are the harmonics
# again).  BILLIARD replaces each cosine by the triangle wave of
# the same frequency and phase: the trajectory of a billiard ball
# in a cube; billiard knots in a cube are exactly the Lissajous
# knots.
#
# References:
# - M. G. V. Bogle, J. E. Hearst, V. F. R. Jones, L. Stoilov,
#   "Lissajous knots", J. Knot Theory Ramif. 3(2):121-140, 1994.
# - L. H. Kauffman, "Fourier Knots", arXiv:q-alg/9711013, 1997.
# - A. K. Trautwein, "Harmonic Knots", PhD thesis, University of
#   Iowa, 1995.
# - J. Hoste, "Torus knots are Fourier-(1,1,2) knots",
#   arXiv:0708.3590, 2007.
# - P.-V. Koseleff, D. Pecker, "Chebyshev knots", J. Knot Theory
#   Ramif. 20(4):575-593, 2011; arXiv:0812.1089.
# - V. F. R. Jones, J. H. Przytycki, "Lissajous knots and billiard
#   knots", Banach Center Publ. 42:145-163, 1998.
# - P.-V. Koseleff, D. Pecker, "Every knot is a billiard knot",
#   arXiv:1106.5600, 2011.

bl_info = {
    "name": "Harmonic Knots",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Harmonic Knot",
    "description": "Lissajous, Fourier, Chebyshev and billiard "
                   "knots as curves or tube meshes",
    "category": "Add Curve",
}

from math import gcd, pi

import numpy as np

try:                                  # inside the math_art package
    from .curve_frames import welded_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import welded_tube

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# ---------------------------------------------------------------
# curve families (numpy-pure)
# ---------------------------------------------------------------

def _tri(x):
    """Triangle wave matching cos in period, phase and range:
    _tri(0) = 1, _tri(pi) = -1, linear in between."""
    return 2.0 * np.abs(np.mod(x, 2.0 * pi) - pi) / pi - 1.0


def _cheb(n, x):
    """Chebyshev polynomial of the first kind T_n evaluated as a
    plain polynomial (valid outside [-1, 1] too)."""
    return np.polynomial.chebyshev.chebval(
        x, [0.0] * int(n) + [1.0])


def lissajous_points(nx, ny, nz, phx, phy, phz, samples,
                     wave='COS'):
    """Closed Lissajous (wave='COS') or cube-billiard (wave='TRI')
    curve, one full period, samples points, endpoint excluded."""
    f = np.cos if wave == 'COS' else _tri
    t = np.linspace(0.0, 2.0 * pi, samples, endpoint=False)
    return np.stack([f(nx * t + phx), f(ny * t + phy),
                     f(nz * t + phz)], axis=1)


def fourier_points(nx, ny, nz, phx, phy, phz,
                   z2_amp, z2_freq, z2_ph, samples):
    """Fourier-(1,1,2) curve: one cosine in x and y, two in z.
    With z2_amp = 0 this degenerates to a Lissajous knot; with the
    trefoil preset values it realizes the (2,3) torus knot (Hoste's
    theorem that every torus knot is Fourier-(1,1,2))."""
    t = np.linspace(0.0, 2.0 * pi, samples, endpoint=False)
    z = np.cos(nz * t + phz) + z2_amp * np.cos(z2_freq * t + z2_ph)
    return np.stack([np.cos(nx * t + phx),
                     np.cos(ny * t + phy), z], axis=1)


def chebyshev_points(a, b, c, phi, samples):
    """Closed Chebyshev knot C(a, b, c, phi).  The polynomial arc
    (T_a(s), T_b(s), T_c(s + phi)) is traced for s in [-1, 1] with
    Chebyshev-node spacing (s = cos theta, so the coordinates are
    cos(a theta) etc. and sampling is densest at the turning
    points).  All diagram crossings of the polynomial curve occur
    for both parameters in [-1, 1], and the arc is closed by a
    one-bridge return path: short horizontal leads out of the two
    endpoint corners (for coprime a, b no other curve point lies
    on those vertical lines), vertical risers, and a straight
    bridge above the whole diagram -- an overpass, so the closure
    is isotopically trivial and the knot type is the standard
    closure of the long Chebyshev knot."""
    m_arc = max(16, int(samples * 5 // 6))
    th = np.linspace(0.0, pi, m_arc)
    s = np.cos(th)[::-1]                       # -1 .. 1
    P = np.stack([_cheb(a, s), _cheb(b, s),
                  _cheb(c, s + phi)], axis=1)
    A, B = P[-1], P[0]                         # ends s=1, s=-1
    ztop = P[:, 2].max() + 1.0
    out = 0.35
    dA = np.array([np.sign(A[0]) or 1.0, np.sign(A[1]) or 1.0,
                   0.0])
    dB = np.array([np.sign(B[0]) or 1.0, np.sign(B[1]) or 1.0,
                   0.0])
    way = [A, A + out * dA,
           np.array([A[0] + out * dA[0], A[1] + out * dA[1],
                     ztop]),
           np.array([B[0] + out * dB[0], B[1] + out * dB[1],
                     ztop]),
           B + out * dB, B]
    m_cl = max(20, samples - m_arc)
    lens = [np.linalg.norm(q - p) for p, q in zip(way[:-1],
                                                  way[1:])]
    total = sum(lens) or 1.0
    legs = []
    for (p, q, ln) in zip(way[:-1], way[1:], lens):
        k = max(3, int(round(m_cl * ln / total)))
        legs.append(np.linspace(p, q, k, endpoint=False))
    # each leg includes its start point, so the first leg would
    # repeat the arc endpoint A; the last leg stops short of B,
    # which is the first arc point again -- a clean closed loop
    legs[0] = legs[0][1:]
    return np.vstack([P] + legs)


def harmonic_knot_points(family, nx, ny, nz, phx, phy, phz,
                         z2_amp=1.0, z2_freq=2, z2_ph=1.4,
                         samples=1000):
    """Raw closed polyline for the chosen family."""
    if family == 'LISSAJOUS':
        return lissajous_points(nx, ny, nz, phx, phy, phz,
                                samples, 'COS')
    if family == 'BILLIARD':
        return lissajous_points(nx, ny, nz, phx, phy, phz,
                                samples, 'TRI')
    if family == 'FOURIER':
        return fourier_points(nx, ny, nz, phx, phy, phz,
                              z2_amp, z2_freq, z2_ph, samples)
    if family == 'CHEBYSHEV':
        return chebyshev_points(nx, ny, nz, phz, samples)
    raise ValueError(family)


def fit_unit_cube(P):
    """Center P at the origin and scale it uniformly so the
    largest half-extent is 1 (the curve fits the 2 m cube)."""
    P = np.asarray(P, float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    P = P - 0.5 * (lo + hi)
    ext = float(np.abs(P).max())
    return P / ext if ext > 0 else P


# ---------------------------------------------------------------
# diagram verification helpers (used by the self-test): compute
# the knot determinant |Delta(-1)| from the xy-diagram of a
# sampled curve via the Wirtinger coloring matrix.  A minor of
# that matrix has determinant equal to the knot determinant, an
# invariant that certifies the preset knot types (trefoil <=> 3
# on a diagram of < 8 crossings).
# ---------------------------------------------------------------

def _diagram_crossings(P):
    """Transverse crossings of the xy-projection of the closed
    polyline P: list of (t_under_over ...) tuples (ti, tj, over_i)
    with fractional curve parameters, or None if a crossing is
    degenerate (strands too close in z)."""
    n = len(P)
    A = P[:, :2]
    D = np.roll(P[:, :2], -1, 0) - A
    Z, Zn = P[:, 2], np.roll(P[:, 2], -1, 0)
    i_i, j_i = np.triu_indices(n, 2)
    keep = ~((i_i == 0) & (j_i == n - 1))     # wrap-adjacent pair
    i_i, j_i = i_i[keep], j_i[keep]
    Di, Dj = D[i_i], D[j_i]
    den = Di[:, 0] * Dj[:, 1] - Di[:, 1] * Dj[:, 0]
    ok = np.abs(den) > 1e-14
    r = A[j_i] - A[i_i]
    dn = np.where(ok, den, 1.0)
    s = np.where(ok, (r[:, 0] * Dj[:, 1] - r[:, 1] * Dj[:, 0])
                 / dn, -1.0)
    u = np.where(ok, (r[:, 0] * Di[:, 1] - r[:, 1] * Di[:, 0])
                 / dn, -1.0)
    hit = ok & (s >= 0) & (s < 1) & (u >= 0) & (u < 1)
    out = []
    for k in np.nonzero(hit)[0]:
        i, j = int(i_i[k]), int(j_i[k])
        zi = Z[i] + s[k] * (Zn[i] - Z[i])
        zj = Z[j] + u[k] * (Zn[j] - Z[j])
        if abs(zi - zj) < 1e-6:
            return None
        out.append((i + float(s[k]), j + float(u[k]), zi > zj))
    return out


def _int_det(M):
    """Exact integer determinant (Bareiss elimination)."""
    M = [row[:] for row in M]
    n = len(M)
    if n == 0:
        return 1
    prev, sign = 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            for r in range(k + 1, n):
                if M[r][k]:
                    M[k], M[r] = M[r], M[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k]
                           - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[-1][-1]


def knot_determinant(P):
    """(determinant, n_crossings) of the sampled closed curve, or
    (None, None) on a degenerate diagram.  Arcs run between
    consecutive underpasses; each crossing contributes the mod-p
    coloring row 2*over - under_in - under_out, and any cofactor
    of that matrix is +-Delta(-1)."""
    cr = _diagram_crossings(P)
    if cr is None:
        return None, None
    nc = len(cr)
    if nc == 0:
        return 1, 0
    ev = []
    for k, (ti, tj, over_i) in enumerate(cr):
        ev.append((ti, k, over_i))
        ev.append((tj, k, not over_i))
    ev.sort()
    over_arc, under_arc = {}, {}
    cur = 0
    for _t, k, over in ev:
        if over:
            over_arc[k] = cur
        else:
            under_arc[k] = cur
            cur += 1
    na = cur                                   # == nc
    M = [[0] * na for _ in range(nc)]
    for k in range(nc):
        M[k][over_arc[k] % na] += 2
        M[k][under_arc[k] % na] -= 1
        M[k][(under_arc[k] + 1) % na] -= 1
    return abs(_int_det([row[:-1] for row in M[:-1]])), nc


# ---------------------------------------------------------------
# presets: parameter sets verified by the self-test below (the
# knot determinants are computed from the sampled curves, so the
# advertised knot types are checked, not assumed)
# ---------------------------------------------------------------

# family, (nx ny nz), (phx phy phz), (z2_amp z2_freq z2_ph)
PRESETS = {
    # classic frequency triple of Bogle-Hearst-Jones-Stoilov;
    # phases 0.7 / 0.2 / 0 give a nontrivial Lissajous knot with
    # determinant 7 (7 crossings in the xy-diagram)
    'LISSAJOUS_327': ('LISSAJOUS', (3, 2, 7), (0.7, 0.2, 0.0),
                      (1.0, 2, 1.4)),
    # Fourier-(1,1,2) trefoil: x = cos 2t, y = cos(3t + 0.15),
    # z = cos(t + 0.7) + cos(2t + 1.4).  Phases fitted
    # numerically; the self-test certifies determinant 3 on a
    # 7-crossing diagram, i.e. the (2,3) torus knot (Hoste)
    'FOURIER_TREFOIL': ('FOURIER', (2, 3, 1), (0.0, 0.15, 0.7),
                        (1.0, 2, 1.4)),
    # harmonic knot H(3,4,5) of Koseleff-Pecker: the trefoil
    'CHEBYSHEV_TREFOIL': ('CHEBYSHEV', (3, 4, 5),
                          (0.0, 0.0, 0.0), (1.0, 2, 1.4)),
    # triangle-wave twin of the Lissajous preset: the trajectory
    # of a billiard ball in a cube (also determinant 7)
    'BILLIARD_327': ('BILLIARD', (3, 2, 7), (0.7, 0.2, 0.0),
                     (1.0, 2, 1.4)),
}


def _pairwise_coprime(a, b, c):
    return gcd(a, b) == 1 and gcd(b, c) == 1 and gcd(a, c) == 1


if _IN_BLENDER:

    def _apply_preset(self, context):
        pr = PRESETS.get(self.preset)
        if pr is None:
            return
        fam, (nx, ny, nz), (px, py, pz), (za, zf, zp) = pr
        self.family = fam
        self.nx, self.ny, self.nz = nx, ny, nz
        self.phx, self.phy, self.phz = px, py, pz
        self.z2_amp, self.z2_freq, self.z2_ph = za, zf, zp

    class CURVE_OT_harmonic_knot_add(bpy.types.Operator):
        """Add a harmonic knot -- a Lissajous, Fourier, Chebyshev
        or cube-billiard curve -- as a curve or tube mesh"""
        bl_idname = "curve.harmonic_knot_add"
        bl_label = "Harmonic Knot"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset",
            description="Ready-made harmonic knot; Custom keeps the "
                        "values below",
            items=[('LISSAJOUS_327', "Lissajous (3,2,7)",
                    "classic Lissajous knot, phases 0.7 / 0.2 / 0"),
                   ('FOURIER_TREFOIL', "Fourier (1,1,2) Trefoil",
                    "the (2,3) torus knot as a Fourier-(1,1,2) "
                    "knot (Hoste)"),
                   ('CHEBYSHEV_TREFOIL', "Chebyshev Trefoil "
                    "(3,4,5)",
                    "harmonic knot H(3,4,5) of Koseleff-Pecker"),
                   ('BILLIARD_327', "Billiard (3,2,7)",
                    "cube billiard trajectory, the triangle-wave "
                    "Lissajous"),
                   ('CUSTOM', "Custom", "keep the values below")],
            default='LISSAJOUS_327', update=_apply_preset)
        family: EnumProperty(
            name="Family",
            description="Which harmonic curve family to trace",
            items=[('LISSAJOUS', "Lissajous",
                    "cosine in each coordinate"),
                   ('FOURIER', "Fourier",
                    "cosine sums; second z-harmonic gives the "
                    "Fourier-(1,1,2) knots"),
                   ('CHEBYSHEV', "Chebyshev",
                    "polynomial knot in Chebyshev polynomials, "
                    "closed by a bridge above the diagram"),
                   ('BILLIARD', "Billiard",
                    "triangle waves: billiard ball in a cube")],
            default='LISSAJOUS')
        nx: IntProperty(
            name="nx / a", default=3, min=1, max=32,
            description="x frequency (Chebyshev: degree a)")
        ny: IntProperty(
            name="ny / b", default=2, min=1, max=32,
            description="y frequency (Chebyshev: degree b)")
        nz: IntProperty(
            name="nz / c", default=7, min=1, max=32,
            description="z frequency (Chebyshev: degree c)")
        phx: FloatProperty(
            name="Phase X", default=0.7, min=-10.0, max=10.0,
            description="x phase in radians (unused by Chebyshev)")
        phy: FloatProperty(
            name="Phase Y", default=0.2, min=-10.0, max=10.0,
            description="y phase in radians (unused by Chebyshev)")
        phz: FloatProperty(
            name="Phase Z", default=0.0, min=-10.0, max=10.0,
            description="z phase in radians (Chebyshev: argument "
                        "shift phi of T_c(s + phi))")
        z2_amp: FloatProperty(
            name="Z Harmonic 2 Amp", default=1.0, min=0.0,
            max=4.0,
            description="Fourier only: amplitude of the second "
                        "z cosine (0 falls back to Lissajous)")
        z2_freq: IntProperty(
            name="Z Harmonic 2 Freq", default=2, min=1, max=32,
            description="Fourier only: frequency of the second "
                        "z cosine")
        z2_ph: FloatProperty(
            name="Z Harmonic 2 Phase", default=1.4, min=-10.0,
            max=10.0,
            description="Fourier only: phase of the second "
                        "z cosine")
        samples: IntProperty(
            name="Samples", default=1000, min=100, max=8000,
            description="Points along the curve")
        output: EnumProperty(
            name="Output",
            description="Curve type to build, or a swept tube mesh",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", ""),
                   ('NURBS', "NURBS Curve", ""),
                   ('MESH', "Mesh Tube", "swept tube mesh")],
            default='BEZIER')
        radius: FloatProperty(
            name="Tube Radius", default=0.05, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius")
        resolution: IntProperty(name="Bevel Resolution",
                                default=6, min=1, max=16,
                                description="Smoothness of the round "
                                            "bevel along the curve")
        tube_sides: IntProperty(name="Tube Sides", default=12,
                                min=3, max=32,
                                description="Number of sides around the "
                                            "swept tube")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the result")

        def execute(self, context):
            if (self.family != 'FOURIER'
                    and not _pairwise_coprime(self.nx, self.ny,
                                              self.nz)):
                self.report(
                    {'WARNING'},
                    f"({self.nx},{self.ny},{self.nz}) not "
                    "pairwise coprime: curve may retrace or "
                    "self-intersect (built anyway)")
            if (self.family in {'LISSAJOUS', 'BILLIARD'}
                    and abs(self.phx) < 1e-9
                    and abs(self.phy) < 1e-9
                    and abs(self.phz) < 1e-9):
                self.report(
                    {'WARNING'},
                    "all phases zero: the curve retraces itself "
                    "(built anyway)")
            P = harmonic_knot_points(
                self.family, self.nx, self.ny, self.nz,
                self.phx, self.phy, self.phz,
                self.z2_amp, self.z2_freq, self.z2_ph,
                self.samples)
            P = fit_unit_cube(P) * self.scale
            fam = self.family.capitalize()
            name = (f"{fam} Knot "
                    f"({self.nx},{self.ny},{self.nz})")
            if self.output == 'MESH':
                verts, faces = welded_tube(
                    P, self.radius, self.tube_sides)
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set(
                    'use_smooth', [True] * len(me.polygons))
                me.update()
                obj = bpy.data.objects.new(name, me)
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                if self.output == 'BEZIER':
                    sp = cu.splines.new('BEZIER')
                    sp.bezier_points.add(len(P) - 1)
                    for i, pnt in enumerate(P):
                        bp = sp.bezier_points[i]
                        bp.co = pnt
                        bp.handle_left_type = 'AUTO'
                        bp.handle_right_type = 'AUTO'
                else:
                    sp = cu.splines.new(self.output)
                    sp.points.add(len(P) - 1)
                    for i, pnt in enumerate(P):
                        sp.points[i].co = (pnt[0], pnt[1],
                                           pnt[2], 1.0)
                    if self.output == 'NURBS':
                        sp.order_u = 4
                sp.use_cyclic_u = True
                cu.bevel_depth = self.radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: {len(P)} samples")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'family')
            for k in ('nx', 'ny', 'nz'):
                lay.prop(self, k)
            if self.family == 'CHEBYSHEV':
                lay.prop(self, 'phz', text="Shift phi")
            else:
                for k in ('phx', 'phy', 'phz'):
                    lay.prop(self, k)
            if self.family == 'FOURIER':
                for k in ('z2_amp', 'z2_freq', 'z2_ph'):
                    lay.prop(self, k)
            lay.prop(self, 'samples')
            lay.prop(self, 'output')
            lay.prop(self, 'radius')
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            elif self.radius > 0:
                lay.prop(self, 'resolution')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.harmonic_knot_add",
                             icon='FORCE_HARMONIC')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(CURVE_OT_harmonic_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_harmonic_knot_add)


def _selftest():
    ok_all = True

    def check(label, cond):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print(f"  {label}: {'OK' if cond else 'BAD'}")

    for key, (fam, nf, ph, z2) in PRESETS.items():
        P = harmonic_knot_points(fam, *nf, *ph, *z2, 900)
        Pn = fit_unit_cube(P)
        seg = np.linalg.norm(np.roll(Pn, -1, 0) - Pn, axis=1)
        print(f"{key} [{fam}] n={len(Pn)}")
        check("finite", np.isfinite(Pn).all())
        # closed: the wrap gap is an ordinary sampling step
        check("closed", seg[-1] < 6.0 * np.median(seg))
        check("consecutive points distinct", seg.min() > 0)
        ext = Pn.max(axis=0) - Pn.min(axis=0)
        check("fits 2 m cube (max extent = 2*scale)",
              abs(ext.max() - 2.0) < 1e-9
              and ext.max() <= 2.0 + 1e-9)
    # knot-type certificates: determinant |Delta(-1)| from the
    # sampled diagram.  Trefoil <=> det 3 on a diagram with
    # fewer than 8 crossings (the only other det-3 knot below
    # 12 crossings is 8_19, which needs >= 8)
    for key, want in (('LISSAJOUS_327', 7),
                      ('BILLIARD_327', 7),
                      ('FOURIER_TREFOIL', 3),
                      ('CHEBYSHEV_TREFOIL', 3)):
        fam, nf, ph, z2 = PRESETS[key]
        P = harmonic_knot_points(fam, *nf, *ph, *z2, 1400)
        det, nc = knot_determinant(P)
        print(f"{key}: determinant={det} crossings={nc}")
        check(f"determinant == {want}", det == want)
        if want == 3:
            check("diagram < 8 crossings (so a trefoil)",
                  nc is not None and nc < 8)
    # Fourier-(1,1,2) torus preset vs. the true (2,3) torus
    # knot: the preset is a different embedding of the same
    # knot, so compare topologically -- the standard torus
    # curve must yield the same determinant certificate (3,
    # on a small diagram) as the preset checked above
    t = np.linspace(0, 2 * pi, 1400, endpoint=False)
    r = 0.7 + 0.3 * np.cos(3 * t)
    T23 = np.stack([r * np.cos(2 * t), r * np.sin(2 * t),
                    0.3 * np.sin(3 * t)], axis=1)
    det_t, nc_t = knot_determinant(T23)
    print(f"true torus (2,3): determinant={det_t} "
          f"crossings={nc_t}")
    check("fourier preset matches torus (2,3) certificate",
          det_t == 3 and nc_t is not None and nc_t < 8)
    # mesh tube via the shared welded sweep
    P = fit_unit_cube(harmonic_knot_points(
        *PRESETS['LISSAJOUS_327'][0:1],
        *PRESETS['LISSAJOUS_327'][1],
        *PRESETS['LISSAJOUS_327'][2],
        *PRESETS['LISSAJOUS_327'][3], 400))
    verts, faces = welded_tube(P, 0.05, 10)
    va = np.asarray(verts)
    print(f"tube: {len(verts)} verts, {len(faces)} faces")
    check("tube non-empty", len(verts) == 400 * 10
          and len(faces) == 400 * 10)
    check("tube finite", np.isfinite(va).all())
    check("tube all quads",
          all(len(f) == 4 for f in faces))
    print("RESULT: " + ("OK" if ok_all else "FAIL"))
    assert ok_all
