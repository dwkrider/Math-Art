
# Petal Knot generator for Blender: knots from PETAL PROJECTIONS
# ("petaluma" knots) -- diagrams with a single multi-crossing.
#
# A petal projection (Adams et al.) is a knot diagram with exactly one
# crossing point, through which the cord passes 2k+1 times, and 2k+1
# non-nested loops ("petals") radiating from it like a daisy.  Every
# knot admits such a projection; its minimal number of petals is the
# knot's PETAL NUMBER.  Because all strands cross at one point, the
# knot is determined ENTIRELY by the permutation giving the heights at
# which the strands pass over one another -- the whole of knot theory
# compressed into a single permutation of {1, ..., 2k+1}.
#
# Construction used here: the planar shadow is the rose curve
#     r = cos(N * theta),  theta in [0, pi),  N = 2k+1 odd,
# whose N petals are equally spaced and whose N passes through the
# origin run along straight lines at angles (2j+1)*pi/(2N).  The
# traversal visits the petal tips with angular stride (N+1)/2, which
# is always coprime to N, so the shadow is a SINGLE closed cord.  The
# j-th pass through the centre is lifted to height ~ perm[j] with the
# C1 profile z = h_j * sin^2(N*theta), returning to the base plane at
# every petal tip; distinct heights make the space curve embedded and
# its shadow the petal diagram.
#
# The preset permutations below were derived and verified for exactly
# this geometry: the lifted curve is projected along a generic tilted
# direction (resolving the multi-crossing into C(N,2) transverse
# crossings), the Alexander polynomial of the resulting diagram is
# evaluated at many integer points by exact integer elimination, and
# the values are matched against the target knot's polynomial up to
# the +-t^a unit.  Exhausting all 5040 seven-petal permutations this
# way reproduces the published classification: with 7 petals exactly
# the knots 0_1, 3_1, 4_1, 5_1, 5_2 and 8_19 occur.  The self-test at
# the bottom re-checks the knot determinant |Delta(-1)| of every
# preset from scratch.
#
# References:
# - Colin Adams, Thomas Crawford, Benjamin DeMeo, Michael Landry,
#   Alex Tong Lin, MurphyKate Montee, Seojung Park, Saraswathi
#   Venkatesh, Farrah Yhee, "Knot projections with a single
#   multi-crossing", Journal of Knot Theory and Its Ramifications
#   24(3), 1550011, 2015 (arXiv:1208.5742) -- petal projections and
#   the petal number.
# - Chaim Even-Zohar, Joel Hass, Nathan Linial, Tahl Nowik,
#   "Invariants of random knots and links", Discrete & Computational
#   Geometry 56(2), 274-314, 2016 (arXiv:1411.3308) -- the Petaluma
#   model: random knots from random petal permutations.
# - Chaim Even-Zohar, "Models of random knots", Journal of Applied
#   and Computational Topology 1, 263-296, 2017 -- survey, including
#   the petal-permutation model.

bl_info = {
    "name": "Petal Knots",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Petal Knot",
    "description": "Knots from petal projections (single "
                   "multi-crossing diagrams): the knot given purely "
                   "by a height permutation",
    "category": "Add Curve",
}

import re
from math import gcd, pi

import numpy as np

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# name: (petals, height permutation, knot determinant |Delta(-1)|,
#        label).  Permutations verified as described in the header.
PRESETS = {
    'UNKNOT':      (5, (1, 2, 3, 4, 5), 1, "Unknot"),
    'TREFOIL':     (5, (1, 3, 5, 2, 4), 3, "Trefoil 3_1"),
    'FIGURE8':     (7, (1, 3, 5, 2, 7, 4, 6), 5, "Figure Eight 4_1"),
    'CINQUEFOIL':  (7, (1, 3, 7, 4, 6, 2, 5), 5, "Cinquefoil 5_1"),
    'THREE_TWIST': (7, (1, 3, 6, 4, 7, 2, 5), 7, "Three-Twist 5_2"),
    'T34':         (7, (1, 4, 7, 3, 6, 2, 5), 3,
                    "Torus Knot T(3,4) = 8_19"),
    'SEPTAFOIL':   (9, (1, 7, 3, 6, 4, 8, 2, 9, 5), 7,
                    "Septafoil 7_1"),
    'T45':         (9, (1, 5, 9, 4, 8, 3, 7, 2, 6), 5,
                    "Torus Knot T(4,5) = 10_124"),
}


def pass_heights(petals, perm, spread=0.35):
    """Height of each of the N passes through the centre: perm mapped
    affinely onto [-spread, +spread]."""
    N = int(petals)
    p = np.asarray(perm, float)
    return spread * (2.0 * p - (N + 1)) / max(N - 1, 1)


def petal_knot_points(petals, perm, samples=64, radius=1.0,
                      spread=0.35):
    """Closed petal-knot space curve as an (N*samples, 3) array (last
    point NOT repeating the first).  Shadow: rose r = cos(N theta);
    pass j through the centre lifted to pass_heights()[j] via
    z = h_j sin^2(N theta), zero (with zero slope) at every petal
    tip.  The result is centred so its bounding box is symmetric
    about the origin; with radius <= 1 and spread <= 1 it fits the
    2 m cube."""
    N = int(petals)
    M = N * int(samples)
    th = np.linspace(0.0, pi, M, endpoint=False)
    r = np.cos(N * th)
    x = radius * r * np.cos(th)
    y = radius * r * np.sin(th)
    j = np.minimum((N * th / pi).astype(int), N - 1)
    h = pass_heights(N, perm, spread)
    z = h[j] * np.sin(N * th) ** 2
    P = np.stack([x, y, z], axis=1)
    P -= 0.5 * (P.min(axis=0) + P.max(axis=0))
    return P


def parse_permutation(text):
    """Parse '1 3 5 2 4' / '1,3,5,2,4' into a height permutation.
    Raises ValueError unless it is a permutation of 1..N, N odd."""
    toks = [t for t in re.split(r"[,;\s]+", str(text).strip()) if t]
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        raise ValueError("permutation entries must be integers")
    N = len(vals)
    if N < 3 or N % 2 == 0:
        raise ValueError("need an odd number of entries (>= 3), "
                         f"got {N}")
    if sorted(vals) != list(range(1, N + 1)):
        raise ValueError(f"not a permutation of 1..{N}")
    return vals


def random_permutation(petals, seed=0):
    """Seeded uniformly random height permutation (Petaluma model)."""
    N = int(petals)
    rng = np.random.default_rng(seed)
    return [int(v) + 1 for v in rng.permutation(N)]


# ----- diagram / determinant machinery (used by the self-test) -----

def _shadow_crossings(P, ta=0.017, tb=0.009):
    """Crossings of the diagram got by projecting the closed polyline
    P along the direction (ta, tb, 1): a list of (s_under, s_over)
    fractional positions along the curve.  For a petal curve the
    single multi-crossing resolves into exactly C(N,2) crossings."""
    X = P[:, 0] - ta * P[:, 2]
    Y = P[:, 1] - tb * P[:, 2]
    Z = P[:, 2]
    Q = np.stack([X, Y], axis=1)
    M = len(Q)
    D = np.roll(Q, -1, 0) - Q
    dZ = np.roll(Z, -1) - Z
    ii, jj = np.triu_indices(M, 2)
    keep = ~((ii == 0) & (jj == M - 1))
    ii, jj = ii[keep], jj[keep]
    Di, Dj = D[ii], D[jj]
    w0 = Q[jj, 0] - Q[ii, 0]
    w1 = Q[jj, 1] - Q[ii, 1]
    den = Di[:, 0] * Dj[:, 1] - Di[:, 1] * Dj[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (w0 * Dj[:, 1] - w1 * Dj[:, 0]) / den
        u = (w0 * Di[:, 1] - w1 * Di[:, 0]) / den
    hit = ((den != 0) & (s >= 0.0) & (s < 1.0)
           & (u >= 0.0) & (u < 1.0))
    out = []
    for k in np.nonzero(hit)[0]:
        i, j = int(ii[k]), int(jj[k])
        si, uj = float(s[k]), float(u[k])
        zi = Z[i] + si * dZ[i]
        zj = Z[j] + uj * dZ[j]
        if zi > zj:
            out.append((j + uj, i + si))
        else:
            out.append((i + si, j + uj))
    return out


def _alexander_det_m1(crossings):
    """Knot determinant |Delta(-1)| from the crossings, via the
    Alexander matrix at t = -1 (where the crossing-sign convention
    drops out: every row is -in - out + 2*over on the Wirtinger
    arcs) and exact integer Bareiss elimination."""
    import bisect
    c = len(crossings)
    unders = sorted(su for su, _ in crossings)
    mat = [[0] * c for _ in range(c)]
    for row, (su, so) in enumerate(crossings):
        k = bisect.bisect_left(unders, su)
        a_in = (k - 1) % c
        a_out = k
        a_over = (bisect.bisect_right(unders, so) - 1) % c
        mat[row][a_in] += -1
        mat[row][a_out] += -1
        mat[row][a_over] += 2
    m = [row[:-1] for row in mat[:-1]]
    n = len(m)
    if n == 0:
        return 1
    sign, prev = 1, 1
    for k in range(n - 1):
        if m[k][k] == 0:
            for r in range(k + 1, n):
                if m[r][k] != 0:
                    m[k], m[r] = m[r], m[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k]
                           - m[i][k] * m[k][j]) // prev
        prev = m[k][k]
    return abs(sign * m[-1][-1])


if _IN_BLENDER:

    _PRESET_ITEMS = [
        (key, PRESETS[key][3],
         f"{PRESETS[key][3]} from {PRESETS[key][0]} petals, "
         f"heights {PRESETS[key][1]}")
        for key in PRESETS
    ] + [
        ('RANDOM', "Random (Petaluma)",
         "Seeded uniformly random height permutation -- a random "
         "knot in the Petaluma model"),
        ('CUSTOM', "Custom Permutation",
         "Type the height permutation yourself"),
    ]

    class CURVE_OT_petal_knot_add(bpy.types.Operator):
        """Add a petal (petaluma) knot: 2k+1 petals around a single
        multi-crossing, the knot type given by the permutation of
        strand heights through the centre"""
        bl_idname = "curve.petal_knot_add"
        bl_label = "Petal Knot"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Knot", items=_PRESET_ITEMS, default='TREFOIL',
            description="Named petal permutation, a seeded random "
                        "one, or your own")
        petals: IntProperty(
            name="Petals", default=7, min=3, max=21,
            description="Number of petals 2k+1 for the random model "
                        "(evens are rounded up to odd)")
        permutation: StringProperty(
            name="Permutation", default="1 3 5 2 4",
            description="Custom heights of the passes through the "
                        "centre, e.g. '1 3 5 2 4': a permutation of "
                        "1..N, N odd; N sets the petal count")
        seed: IntProperty(
            name="Seed", default=0, min=0,
            description="Random-permutation seed")
        petal_radius: FloatProperty(
            name="Petal Radius", default=1.0, min=0.05, max=10.0,
            description="Radius of the circle of petal tips")
        spread: FloatProperty(
            name="Height Spread", default=0.35, min=0.02, max=2.0,
            description="Half-range of the strand heights at the "
                        "centre; raise it (or thin the tube) for "
                        "many petals")
        samples: IntProperty(
            name="Samples / Petal", default=64, min=8, max=512,
            description="Curve samples per petal")
        output: EnumProperty(
            name="Output",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", ""),
                   ('NURBS', "NURBS Curve", ""),
                   ('MESH', "Mesh Tube", "welded swept tube mesh")],
            default='BEZIER')
        radius: FloatProperty(
            name="Tube Radius", default=0.035, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius (keep "
                        "below half the centre height gap "
                        "spread/(N-1))")
        resolution: IntProperty(name="Bevel Resolution", default=6,
                                min=1, max=16)
        tube_sides: IntProperty(name="Tube Sides", default=12,
                                min=3, max=32)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            if self.preset in PRESETS:
                N, perm, _det, label = PRESETS[self.preset]
            elif self.preset == 'RANDOM':
                N = min(self.petals | 1, 21)
                perm = random_permutation(N, self.seed)
                label = f"Random {N}p #{self.seed}"
            else:                                        # CUSTOM
                try:
                    perm = parse_permutation(self.permutation)
                except ValueError as e:
                    self.report({'ERROR'}, f"Permutation: {e}")
                    return {'CANCELLED'}
                N = len(perm)
                label = f"Custom {N}p"
            P = petal_knot_points(N, perm, self.samples,
                                  self.petal_radius, self.spread)
            P = P * self.scale
            name = f"Petal Knot {label}"
            if self.output == 'MESH':
                try:
                    from . import knot_carpet_generator as kcg
                except ImportError:
                    import knot_carpet_generator as kcg
                verts, faces = kcg._tube_welded(
                    P, max(self.radius, 1e-3), self.tube_sides)
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
                        sp.points[i].co = (pnt[0], pnt[1], pnt[2],
                                           1.0)
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
            self.report(
                {'INFO'},
                f"{name}: {N} petals, heights "
                f"{tuple(perm)}, {N * (N - 1) // 2} crossings at "
                f"the centre")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'CUSTOM':
                lay.prop(self, 'permutation')
            elif self.preset == 'RANDOM':
                lay.prop(self, 'petals')
                lay.prop(self, 'seed')
            for k in ('petal_radius', 'spread', 'samples', 'output',
                      'radius'):
                lay.prop(self, k)
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            elif self.radius > 0:
                lay.prop(self, 'resolution')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.petal_knot_add",
                             icon='CURVE_NCIRCLE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(CURVE_OT_petal_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_petal_knot_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        try:
            from . import knot_carpet_generator as kcg
        except ImportError:
            import knot_carpet_generator as kcg
        ok_all = True

        def _chk(cond, msg):
            global ok_all
            print(f"  {'OK ' if cond else 'BAD'} {msg}")
            ok_all = ok_all and bool(cond)

        for key, (N, perm, det_want, label) in PRESETS.items():
            print(f"{key} ({label}): {N} petals, heights {perm}")
            _chk(N % 2 == 1 and sorted(perm) == list(range(1, N + 1)),
                 f"valid odd permutation of 1..{N}")
            stride = (N + 1) // 2
            orbit = {(m * stride) % N for m in range(N)}
            _chk(gcd(stride, N) == 1 and len(orbit) == N,
                 f"tip stride {stride} is a single {N}-cycle "
                 "(one closed cord)")
            h = pass_heights(N, perm)
            _chk(len(set(np.round(h, 12))) == N
                 and np.array_equal(np.argsort(np.argsort(h)) + 1,
                                    np.asarray(perm)),
                 f"{N} distinct centre heights ranked as the "
                 "permutation")
            P = petal_knot_points(N, perm, samples=64)
            d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
            _chk(np.all(np.isfinite(P)) and d.min() > 1e-9,
                 f"finite closed curve, {len(P)} pts, consecutive "
                 "points distinct (min step "
                 f"{d.min():.2e})")
            ext = P.max(axis=0) - P.min(axis=0)
            ctr = 0.5 * (P.max(axis=0) + P.min(axis=0))
            _chk(np.all(ext <= 2.0 + 1e-9)
                 and np.all(np.abs(ctr) < 1e-9),
                 f"centred, fits the 2 m cube (extent "
                 f"{ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f})")
            cr = _shadow_crossings(P)
            want = N * (N - 1) // 2
            _chk(len(cr) == want,
                 f"resolved multi-crossing has C({N},2) = {want} "
                 "crossings")
            det = _alexander_det_m1(cr)
            _chk(det == det_want,
                 f"knot determinant |Delta(-1)| = {det} "
                 f"(expect {det_want})")

        # random model + custom parser
        for N, seed in ((9, 1), (11, 7)):
            perm = random_permutation(N, seed)
            _chk(sorted(perm) == list(range(1, N + 1)),
                 f"random_permutation({N}, seed={seed}) valid")
            P = petal_knot_points(N, perm, samples=32)
            _chk(len(_shadow_crossings(P)) == N * (N - 1) // 2,
                 f"random {N}-petal curve: C(N,2) crossings")
        _chk(parse_permutation("1, 3;5 2\t4") == [1, 3, 5, 2, 4],
             "parse_permutation accepts mixed separators")
        for bad in ("1 2 3 4", "1 2 2 4 5", "1 2 x", ""):
            try:
                parse_permutation(bad)
                _chk(False, f"parse_permutation rejects {bad!r}")
            except ValueError:
                _chk(True, f"parse_permutation rejects {bad!r}")

        # mesh tube via the knot-carpet welded sweep
        N, perm, _dw, _lb = PRESETS['TREFOIL']
        P = petal_knot_points(N, perm, samples=64)
        verts, faces = kcg._tube_welded(P, 0.03, 8)
        V = np.asarray(verts, float)
        _chk(len(verts) == len(P) * 8 and len(faces) == len(P) * 8
             and np.all(np.isfinite(V))
             and all(len(f) == 4 for f in faces),
             f"mesh tube: {len(verts)} verts, {len(faces)} faces, "
             "finite, all quads")

        print("RESULT: " + ("OK" if ok_all else "FAILED"))
        assert ok_all
