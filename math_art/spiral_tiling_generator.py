
# Logarithmic Spiral Tiling of Triangles -- generator for Blender
#
# Robert Fathauer's logarithmic spiral tilings of triangles ("Logarithmic
# Spiral Tilings of Triangles", Bridges 2021, pp. 55-62), produced as
# finite annular patches of colored triangle tiles fed through the shared
# Pattern Engine (see the `patterns` package) for color, grout margin and
# relief.
#
# THE PROTOTILE.  A single triangle with interior angles A, B, C and
# opposite side lengths a, b, c.  The side c (opposite C) is set to 1
# without loss of generality, so by the Law of Sines
#
#       a = sin(A)/sin(C),   b = sin(B)/sin(C),   c = 1.
#
# SELF-SIMILAR SPIRAL.  Adjacent triangles scale by a constant factor s
# in (0, 1), share a vertex, and mate along an edge, winding around a
# central singular point O where the tiles become infinitesimally small.
# The whole tiling is the orbit of ONE prototile under a spiral
# similarity  T  (scale s, rotation A) whose fixed point is O.  Following
# Fathauer's Figure 3: the next (s-scaled) triangle places its C-corner
# on the previous triangle's B-corner and lays its b-edge (length b*s)
# along the previous triangle's unit c-edge; after n such reduced
# triangles the spiral has wound once around O and the n-th triangle's
# a-edge (length a*s^n) lies on the remaining part of that same c-edge.
# Equating the two pieces of the unit side gives Fathauer's Eq. (1):
#
#       a*s^n + b*s = 1,                                            (1)
#
# solved numerically for the real root s in (0, 1).  Here n is the number
# of reduced triangles mated before one shares an edge with the start.
# Because the n-th tile must rotate through A n times to share that edge,
# and a further B to regain the starting orientation, n*A + B = 360/m,
# where m is the number of spiral arms; with B = 180 - A - C this is
# Fathauer's Eq. (3):
#
#       A = (360/m - 180 + C)/(n - 1)                              (3)
#
# (single arm m = 1 reduces to Eq. (2), A = (180 + C)/(n - 1)).  The
# controlling angle A must satisfy 180/(n-1) < A < 360/n for m = 1.  The
# tiling built here is the union of the single-arm orbit and its m-1
# copies rotated by 360/m about O; a single-arm spiral already reads as n
# counter-rotating arms, and an m-armed spiral shows m*n arms in the
# opposite direction.
#
# NAMED FAMILIES (each fixes the prototile angles):
#   Equilateral       A=B=C=60, admitted only for n=5; s solves
#                     s^5 + s = 1 (inverse plastic number ~ 0.7549).
#   Golden            C=36, n=3 -> A=108; a = golden ratio, s = 1/phi.
#   Isosceles side-to-side (B=C):  C = 180*(n-2)/(2n-1),  n>2.
#   Isosceles side-to-base (A=B):  C = 180*(n-3)/(n+1),   n>3.
#   Isosceles base-to-side (A=C):  C = 180/(n-2),         n>4.
#   Right leg-to-hypotenuse (C=90):  A = 270/(n-1).
#   Right hypotenuse-to-leg (B=90):  C = 90*(n-3)/n,      n>3.
#   Any isosceles     special case m=n=2 gives A=C for a continuous
#                     range of the side angle.
#   Regular m-gon     n=1: a regular m-gon corner gives C = 180 - 360/m
#                     and m-armed spirals for a continuous range of A.
#   General           user picks n, m and the controlling angle C.
#
# References:
#   Robert W. Fathauer, "Logarithmic Spiral Tilings of Triangles",
#     Bridges 2021 Conference Proceedings, pp. 55-62 -- the construction,
#     equations and named families reproduced here.
#   C. Waldman, "Gnomon is an Island" (2016) and private communication --
#     the side-to-side isosceles family (any n > 2) and the observation
#     that these tilings are self-dual.
#   Branko Grunbaum & G. C. Shephard, "Tilings and Patterns" (W. H.
#     Freeman, 1987) -- spiral tilings and their singular points.

bl_info = {
    "name": "Spiral Tiling of Triangles",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Fathauer logarithmic spiral tilings of similar "
                   "triangles: equilateral, golden, isosceles, right, "
                   "and multi-armed families as colored tile patterns",
    "category": "Add Mesh",
}

from math import sin, cos, pi, radians, hypot, atan2

import numpy as np

try:
    from .patterns import common as pc
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    from patterns import common as pc
    import tiling_generator as tg


# --------------------------------------------------------------------
# Prototile angles per family
# --------------------------------------------------------------------
#
# Each family fixes the triangle's angles (A, B, C) from n, m and, for the
# continuous families, a controlling angle.  family_geometry returns the
# resolved (A, B, C, n, m) in degrees; some families override the caller's
# n and/or m to their only admissible value.

def _iso_ss(n):     # side-to-side  (B = C)
    return 180.0 * (n - 2) / (2 * n - 1)


def _iso_sb(n):     # side-to-base  (A = B)
    return 180.0 * (n - 3) / (n + 1)


def _iso_bs(n):     # base-to-side  (A = C)
    return 180.0 / (n - 2)


def family_geometry(family, n, m, angle):
    """Resolve (A, B, C, n, m) in degrees for a family.

    `angle` is the user's controlling angle, used only by the continuous
    families (ANY isosceles, regular m-gon, and General); the fixed
    families ignore it and derive every angle from n (and m).
    """
    n = max(1, int(n))
    m = max(1, int(m))

    if family == 'EQUILATERAL':
        return 60.0, 60.0, 60.0, 5, 1           # admitted only for n = 5

    if family == 'GOLDEN':
        C = 36.0
        n = 3
        A = (360.0 / m - 180.0 + C) / (n - 1)
        return A, 180.0 - A - C, C, n, m

    if family == 'ISO_ANY':                     # special case m = n = 2
        n, m = 2, 2                             # A = C, any isosceles
        C = min(89.0, max(1.0, float(angle)))
        A = C
        return A, 180.0 - A - C, C, n, m

    if family == 'POLY':                        # n = 1, regular m-gon
        n = 1
        C = 180.0 - 360.0 / m
        hi = 360.0 / m
        A = min(hi - 1.0, max(1.0, float(angle)))
        return A, 180.0 - A - C, C, n, m

    if family == 'ISO_SS':
        n = max(3, n)
        C = _iso_ss(n)
    elif family == 'ISO_SB':
        n = max(4, n)
        C = _iso_sb(n)
    elif family == 'ISO_BS':
        n = max(5, n)
        C = _iso_bs(n)
    elif family == 'RIGHT_LH':                  # leg-to-hypotenuse, C = 90
        n = max(5, n)                           # n > 4 keeps B > 0
        C = 90.0
    elif family == 'RIGHT_HL':                  # hypotenuse-to-leg, B = 90
        n = max(4, n)
        C = 90.0 * (n - 3) / n
    elif family == 'GENERAL':
        C = min(179.0, max(1.0, float(angle)))
    else:
        raise ValueError("unknown spiral family %r" % family)

    A = (360.0 / m - 180.0 + C) / (n - 1)
    return A, 180.0 - A - C, C, n, m


# --------------------------------------------------------------------
# Pure-math core (no bpy)
# --------------------------------------------------------------------

def solve_scale(a, b, n, iters=200):
    """Real root s in (0, 1) of Eq. (1)  a*s^n + b*s = 1  by bisection.
    f(s) = a*s^n + b*s - 1 is increasing on (0, 1) with f(0) = -1 < 0, so
    a unique root exists whenever a + b > 1 (always true for a triangle
    with c = 1, since the two other sides exceed the third)."""
    lo, hi = 1e-12, 1.0

    def f(s):
        return a * s ** n + b * s - 1.0

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def prototile_similarity(A, B, C, s):
    """Build the prototile and the spiral similarity T that generates the
    tiling, returning (P, Q, R, alpha, beta, O) as complex numbers.

    Prototile corners (angles at each): R = A-corner at the origin,
    P = C-corner, Q = B-corner.  Edges: PQ = a (opposite A), QR = c = 1
    (opposite C), RP = b (opposite B).  The next tile T(prototile) puts
    its C-corner on Q and its A-corner A1 = Q + b*s*(R - Q) on the unit
    c-edge QR; T is the orientation-preserving similarity with T(P) = Q
    and T(R) = A1 (so |alpha| = s and arg(alpha) = A), fixed point O."""
    Ar = radians(A)
    Cr = radians(C)
    b = sin(radians(B)) / sin(Cr)
    R = 0.0 + 0.0j
    P = b + 0.0j                                # RP = b along +x
    Q = cos(Ar) + 1j * sin(Ar)                  # RQ = c = 1 at angle A
    A1 = Q + b * s * (R - Q)                    # next A-corner on QR
    alpha = (A1 - Q) / (R - P)                  # T(P)=Q, T(R)=A1
    beta = Q - alpha * P
    O = beta / (1.0 - alpha)
    return P, Q, R, alpha, beta, O


def _ring_range(s, rings):
    """(kmin, kmax) scale levels for the orbit: `rings` levels outward
    (bigger) from the prototile, and inward (smaller) far enough that the
    tiles converge visibly to the singular point, capped for sanity."""
    rings = max(1, int(rings))
    if s >= 0.999:
        inner = 40
    else:
        inner = int(np.ceil(np.log(0.004) / np.log(s)))   # s^inner ~ 0.4%
    inner = max(6, min(inner, 80))
    return -rings, inner


def spiral_patch(family, n, m, angle, rings):
    """Return (polys, arms, levels) for a finite spiral-tiling patch.

    polys  -- list of (3, 2) CCW float arrays, one similar triangle each
    arms   -- parallel list: which spiral arm (0..m-1) each tile is on
    levels -- parallel list: the scale-level index k (0 = prototile),
              larger k = smaller/inner tiles
    """
    A, B, C, n, m = family_geometry(family, n, m, angle)
    if min(A, B, C) <= 0.25 or max(A, B, C) >= 179.75:
        raise ValueError("angles A=%.2f B=%.2f C=%.2f are degenerate for "
                         "this n/m; adjust n, arms or angle" % (A, B, C))
    a = sin(radians(A)) / sin(radians(C))
    b = sin(radians(B)) / sin(radians(C))
    s = solve_scale(a, b, n)
    P, Q, R, alpha, beta, O = prototile_similarity(A, B, C, s)

    kmin, kmax = _ring_range(s, rings)
    base_polys, base_levels = [], []
    for k in range(kmin, kmax + 1):
        ak = alpha ** k
        bk = beta * (1.0 - ak) / (1.0 - alpha)
        pts = [ak * z + bk for z in (P, Q, R)]
        base_polys.append(np.array([[z.real, z.imag] for z in pts], float))
        base_levels.append(k - kmin)

    polys, arms, levels = [], [], []
    for j in range(m):
        rot = np.exp(1j * 2.0 * pi * j / m)
        for poly, lev in zip(base_polys, base_levels):
            z = (poly[:, 0] + 1j * poly[:, 1] - O) * rot + O
            polys.append(np.column_stack([z.real, z.imag]))
            arms.append(j)
            levels.append(lev)
    return polys, arms, levels


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

FAMILY_ITEMS = [
    ('EQUILATERAL', "Equilateral (plastic number)",
     "The single equilateral-triangle spiral (n = 5); successive tiles "
     "scale by the inverse plastic number, s^5 + s = 1 ~ 0.7549"),
    ('GOLDEN', "Golden Triangle",
     "The golden-triangle spiral: C = 36, n = 3, apex A = 108; the "
     "long-to-short edge ratio and the scale factor s = 1/phi are both "
     "the golden mean"),
    ('ISO_SS', "Isosceles, side-to-side (B = C)",
     "Isosceles family mating side-to-side, one tiling per n > 2 "
     "(n = 3 is the golden triangle, n = 5 the equilateral)"),
    ('ISO_SB', "Isosceles, side-to-base (A = B)",
     "Isosceles family mating side-to-base, one tiling per n > 3 "
     "(n = 5 is the equilateral)"),
    ('ISO_BS', "Isosceles, base-to-side (A = C)",
     "Isosceles family mating base-to-side, one tiling per n > 4 "
     "(n = 5 is the equilateral)"),
    ('RIGHT_LH', "Right, leg-to-hypotenuse (C = 90)",
     "Right-triangle family with the right angle at C, A = 270/(n-1)"),
    ('RIGHT_HL', "Right, hypotenuse-to-leg (B = 90)",
     "Right-triangle family with the right angle at B, C = 90(n-3)/n, "
     "one tiling per n > 3"),
    ('ISO_ANY', "Any Isosceles (m = n = 2)",
     "The special two-armed case m = n = 2 (A = C), admitted for a "
     "continuous range of the isosceles side angle set by Angle"),
    ('POLY', "Regular m-gon (n = 1)",
     "n = 1: a regular m-gon corner, C = 180 - 360/m, gives m-armed "
     "spirals for a continuous range of the angle A set by Angle"),
    ('GENERAL', "General / Custom",
     "Pick n, arms m and the controlling angle C freely; the remaining "
     "angles follow from A = (360/m - 180 + C)/(n - 1)"),
]

# families whose n is a free control (the rest force n)
_FREE_N = {'ISO_SS', 'ISO_SB', 'ISO_BS', 'RIGHT_LH', 'RIGHT_HL', 'GENERAL'}
# families whose arm count m is a free control
_FREE_M = {'GOLDEN', 'POLY', 'GENERAL',
           'ISO_SS', 'ISO_SB', 'ISO_BS', 'RIGHT_LH', 'RIGHT_HL'}
# families driven by the continuous controlling Angle
_FREE_ANGLE = {'ISO_ANY', 'POLY', 'GENERAL'}


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_spiral_tiling_add(bpy.types.Operator, AddObjectHelper):
        """Add a Fathauer logarithmic spiral tiling of triangles"""
        bl_idname = "mesh.spiral_tiling_add"
        bl_label = "Spiral Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(name="Family", items=FAMILY_ITEMS,
                             default='GOLDEN')
        n: IntProperty(
            name="n", default=3, min=1, max=24,
            description="Number of reduced triangles mated before one "
                        "shares an edge with the starting triangle; sets "
                        "the spiral pitch (fixed families override this)")
        arms: IntProperty(
            name="Arms (m)", default=1, min=1, max=16,
            description="Number of spiral arms in one direction; the "
                        "closure becomes n*A + B = 360/m")
        angle: FloatProperty(
            name="Angle", default=50.0, min=1.0, max=179.0,
            description="Controlling angle in degrees for the continuous "
                        "families: the isosceles side angle C (= A) for "
                        "Any Isosceles, the angle A for the Regular m-gon, "
                        "or the angle C for General")
        rings: IntProperty(
            name="Rings", default=3, min=1, max=10,
            description="How many scale levels to build outward from the "
                        "prototile; the spiral is always carried inward "
                        "far enough to converge on the singular point")
        color_by: EnumProperty(
            name="Color By",
            items=[('ARM', "By Arm",
                    "A material per spiral arm"),
                   ('RING', "By Ring / Size",
                    "A material per scale level, so tiles band into "
                    "concentric rings by size (colors cycle)"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='RING')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat 2D mesh; > 0 extrudes each tile")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object "
                        "(parented to an empty) so tiles can be edited "
                        "individually")

        def execute(self, context):
            try:
                polys, arms, levels = spiral_patch(
                    self.family, self.n, self.arms, self.angle, self.rings)
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            if self.color_by == 'ARM':
                npal = len(pc.PALETTE_RGBA)
                types, cby = [x % npal for x in arms], 'TYPE'
            elif self.color_by == 'RING':
                npal = len(pc.PALETTE_RGBA)
                types, cby = [x % npal for x in levels], 'TYPE'
            else:
                types, cby = levels, 'UNIFORM'
            cells = tg.cells_from_polys(
                lambda a, b: (polys, types), 1, 1, cby,
                self.margin, self.height, False)
            label = dict((k, v) for k, v, _ in FAMILY_ITEMS)[self.family]
            obj = pc.emit(context, "Spiral %s" % label, cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no tiling generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s  V=%d F=%d" %
                            (label, len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s  %d tiles" %
                            (label, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            if self.family in _FREE_N:
                lay.prop(self, 'n')
            if self.family in _FREE_M:
                lay.prop(self, 'arms')
            if self.family in _FREE_ANGLE:
                lay.prop(self, 'angle')
            lay.prop(self, 'rings')
            for p in ('color_by', 'margin', 'height'):
                lay.prop(self, p)
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.spiral_tiling_add", icon='FORCE_VORTEX')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_spiral_tiling_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_spiral_tiling_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _edge_lengths(poly):
    p = np.asarray(poly, float)
    return sorted(float(hypot(*(p[(k + 1) % 3] - p[k]))) for k in range(3))


def _point_on_segment(pt, a, b, tol):
    """True if point pt lies on segment a-b within tol (distance to the
    line and inside the parameter range)."""
    pt = np.asarray(pt, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    L2 = float(d @ d)
    if L2 < 1e-18:
        return hypot(*(pt - a)) < tol
    t = float((pt - a) @ d) / L2
    proj = a + max(0.0, min(1.0, t)) * d
    return hypot(*(pt - proj)) < tol


def _coverage_annulus(polys, O, r_in, r_out, n_rad=40, n_ang=180,
                      off=(0.0031, 0.0017)):
    """Dense annulus sampling around the singular point O.  Returns
    (total, gaps, overlaps): points on no tile / on two-or-more tiles."""
    gaps = overs = total = 0
    for ir in range(n_rad):
        r = r_in + (r_out - r_in) * (ir + 0.5) / n_rad
        for ia in range(n_ang):
            th = 2.0 * pi * (ia + 0.5) / n_ang
            x = O[0] + r * cos(th) + off[0]
            y = O[1] + r * sin(th) + off[1]
            total += 1
            cnt = 0
            for p in polys:
                if tg._pip(p, x, y):
                    cnt += 1
                    if cnt >= 2:
                        break
            if cnt == 0:
                gaps += 1
            elif cnt >= 2:
                overs += 1
    return total, gaps, overs


def _self_test():
    cases = [
        ("Equilateral      ", 'EQUILATERAL', 5, 1, 60.0),
        ("Golden           ", 'GOLDEN', 3, 1, 36.0),
        ("Iso side-to-side ", 'ISO_SS', 4, 1, 0.0),
        ("Iso side-to-side ", 'ISO_SS', 6, 1, 0.0),
        ("Iso side-to-base ", 'ISO_SB', 6, 1, 0.0),
        ("Iso base-to-side ", 'ISO_BS', 6, 1, 0.0),
        ("Right leg-to-hyp ", 'RIGHT_LH', 5, 1, 0.0),
        ("Right hyp-to-leg ", 'RIGHT_HL', 5, 1, 0.0),
        ("Scalene (Fig 3)  ", 'GENERAL', 5, 1, 80.0),
        ("Two-arm A=C      ", 'ISO_ANY', 2, 2, 50.0),
        ("Multi-arm 3x2    ", 'GENERAL', 2, 3, 80.0),
        ("Multi-arm 2x4sca ", 'GENERAL', 4, 2, 50.0),
        ("Regular 3-gon    ", 'POLY', 1, 3, 60.0),
        ("Regular 4-gon    ", 'POLY', 1, 4, 45.0),
        ("Regular 5-gon    ", 'POLY', 1, 5, 50.0),
    ]
    all_ok = True
    print("%-18s  n  m   A      B      C      s       eq1      "
          "sim  adj  cover(g/o)  tiles" % "family")
    for name, fam, n_in, m_in, ang in cases:
        A, B, C, n, m = family_geometry(fam, n_in, m_in, ang)
        a = sin(radians(A)) / sin(radians(C))
        b = sin(radians(B)) / sin(radians(C))
        s = solve_scale(a, b, n)
        eq1 = a * s ** n + b * s - 1.0
        angsum = abs(A + B + C - 180.0)

        rings_test = n + 3        # generous outer margin for the test band
        polys, arms, levels = spiral_patch(fam, n_in, m_in, ang, rings_test)
        # arm-0 tiles, already emitted in level order (0 = largest ring)
        base = [p for p, j in zip(polys, arms) if j == 0]

        # (i) every tile similar to the prototile, size ratios powers of s
        e0 = _edge_lengths(base[0])
        shape_ok = True
        pow_ok = True
        for p in base:
            e = _edge_lengths(p)
            r = e[2] / e0[2]
            # same shape: sorted edge ratios match the prototile's
            for k in range(3):
                if abs(e[k] / e0[k] - r) > 1e-6:
                    shape_ok = False
            kk = np.log(r) / np.log(s)
            if abs(kk - round(kk)) > 1e-6:
                pow_ok = False
        sim_ok = shape_ok and pow_ok

        # (ii) adjacency: consecutive tiles mate edge-to-edge -- the
        # smaller tile's b-edge (P..R, its C-corner to A-corner) lies on
        # the larger tile's c-edge (Q..R, its B-corner to A-corner)
        adj_ok = True
        tol = 1e-6 * e0[2]
        for i in range(len(base) - 1):
            big = base[i]                       # corners order P,Q,R
            small = base[i + 1]
            cQ, cR = big[1], big[2]             # big c-edge Q..R
            bP, bR = small[0], small[2]         # small b-edge P..R
            if not (_point_on_segment(bP, cQ, cR, tol) and
                    _point_on_segment(bR, cQ, cR, tol)):
                adj_ok = False
                break

        # (iii) gap/overlap-free coverage of an interior annulus centred
        # on the singular point O.  The band sits a few scale levels
        # inside the outermost ring (margin > n on the outer edge, so it
        # never reaches the ragged spiral boundary) and well outside the
        # innermost built tiles.
        Pp, Qp, Rp, _, _, Oc = prototile_similarity(A, B, C, s)
        O = (Oc.real, Oc.imag)
        c0 = (Pp + Qp + Rp) / 3.0
        r_ref = hypot(c0.real - O[0], c0.imag - O[1])   # prototile radius
        total, gaps, overs = _coverage_annulus(
            polys, O, r_ref * s ** 5, r_ref / s)

        ok = (eq1 < 1e-9 and angsum < 1e-9 and sim_ok and adj_ok
              and gaps == 0 and overs == 0)
        all_ok = all_ok and ok
        print("%-18s %2d %2d %6.2f %6.2f %6.2f %.5f %+8.1e  %s   %s  "
              "%5d/%-4d  %5d  %s"
              % (name, n, m, A, B, C, s, eq1,
                 "T" if sim_ok else "F", "T" if adj_ok else "F",
                 gaps, overs, len(polys), "OK" if ok else "BAD"))

    # reproduce the two paper landmarks
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    A_g, B_g, C_g, n_g, m_g = family_geometry('GOLDEN', 3, 1, 36.0)
    a_g = sin(radians(A_g)) / sin(radians(C_g))
    s_g = solve_scale(a_g, sin(radians(B_g)) / sin(radians(C_g)), n_g)
    gold_ok = (abs(A_g - 108.0) < 1e-9 and abs(s_g - 1.0 / phi) < 1e-9
               and abs(a_g - phi) < 1e-9)
    s_e = solve_scale(1.0, 1.0, 5)
    equi_ok = abs(s_e ** 5 + s_e - 1.0) < 1e-12
    print("landmark  GOLDEN A=108 & s=1/phi & a=phi : %s   (A=%.4f "
          "s=%.10f 1/phi=%.10f)" % ("OK" if gold_ok else "BAD",
                                    A_g, s_g, 1.0 / phi))
    print("landmark  EQUILATERAL s^5+s=1            : %s   (s=%.10f "
          "plastic^-1=0.7549)" % ("OK" if equi_ok else "BAD", s_e))
    all_ok = all_ok and gold_ok and equi_ok
    print("RESULT:", "OK" if all_ok else "BAD")
    return all_ok
