
# Hyperbolic Tiling Generator for Blender
#
# The hyperbolic analogue of the Euclidean uniform-tiling generator:
# it goes beyond the regular {p,q} tilings to the whole Wythoff
# family -- truncated, rectified, cantellated, omnitruncated and the
# chiral snub -- built on a general (p, q, r) triangle reflection group,
# and renders each tiling either as flat colored polygons in the unit
# disk or wrapped onto a curved 3D surface model.
#
# Machinery.  Everything lives in the hyperboloid model of H^2 inside
# Minkowski R^{2,1}, <x,y> = x1 y1 + x2 y2 - x3 y3.  The (p, q, r)
# triangle group is three mirrors with unit spacelike normals whose
# pairwise Gram products are the corner angles:
#
#     <n0, n1> = -cos(pi/p)    corner A (mirrors 0,1), order p
#     <n1, n2> = -cos(pi/q)    corner B (mirrors 1,2), order q
#     <n0, n2> = -cos(pi/r)    corner C (mirrors 0,2), order r
#
# hyperbolic exactly when 1/p + 1/q + 1/r < 1 (r defaults to 2, the
# right triangle that gives the classic regular families).  Reflection
# in a mirror is x -> x - 2<x,n>n; the group is enumerated by BFS over
# reflection words to a tile cap.
#
# Wythoff construction.  A uniform tiling is the orbit of a single
# SEED point in the fundamental triangle.  Which of the three mirrors
# are "active" (ringed Coxeter nodes) fixes the seed: it lies on every
# inactive mirror and is equidistant from the active ones (equal-length
# edges -> a uniform, vertex-transitive tiling).  With the linear Gram
# picture this seed is the null direction of two linear constraints,
# renormalized to the hyperboloid:
#
#     REGULAR   {p,q}    rings {0}      seed at corner B
#     TRUNCATED t{p,q}   rings {0,1}    seed on edge BC
#     RECTIFIED r{p,q}   rings {1}      seed at corner C
#     BITRUNC.  t{q,p}   rings {1,2}    seed on edge AB
#     CANTEL.   rr{p,q}  rings {0,2}    seed on edge CA
#     OMNITRUNC tr{p,q}  rings {0,1,2}  seed at the incentre
#
# Faces.  Each corner of order n whose two mirrors are k-of-them active
# contributes a face centred there: an n-gon when k = 1, a 2n-gon when
# k = 2, and no face (that corner is a vertex) when k = 0.  A face is
# the orbit of the seed under the finite dihedral stabilizer of its
# corner -- always the COMPLETE local ring, so no partial faces are
# ever emitted -- pushed out to every image of that corner by the group
# and de-duplicated by centre.  A degenerate 2-gon (an order-2 corner
# with one active mirror, i.e. r = 2) is just an edge and is dropped.
# SNUB is the odd one out (the alternation of the omnitruncation: the
# rotation subgroup only, chiral); see snub_faces.
#
# Every face is kept as its ordered ring of hyperboloid corner points,
# so it can be projected into any output model.  The two flat disk
# pictures go through the shared 2D tiling pipeline (color / grout
# margin / relief / backing); the two curved surfaces are built as 3D
# cells directly:
#   KLEIN     (x,y,t) -> (x/t, y/t).  Geodesics are straight chords, so
#             each face is a straight-edged polygon of its corners.
#   POINCARE  (x,y,t) -> (x/(1+t), y/(1+t)).  Geodesic edges are arcs,
#             so every edge is subdivided by sampling the hyperboloid
#             geodesic between its endpoints before projecting.
#   HEMISPHERE  Klein disk K = (x/t, y/t) lifted onto the upper unit
#             hemisphere z = sqrt(1 - |K|^2); edges subdivided.
#   PSEUDOSPHERE  the conformal disk carried Poincare -> Cayley -> upper
#             half plane -> tractricoid (the isometry y = cosh u, x = v),
#             clipped to one seam period 0 <= x <= 2pi and 1 <= y <= cap;
#             edges subdivided.
#
# Spidron fill (Kaplan's "Hyperbolic Spidrons").  Every face of a
# uniform tiling is a regular hyperbolic polygon, and a spidron nest
# can be drawn inside it by iterating inscribed polygons -- exactly the
# construction the Euclidean spidron generators use, lifted to H^2.
# The whole construction lives in the KLEIN disk, where hyperbolic
# geodesics are straight chords: "connect alternate vertices" is
# literal 2D line intersection, so the star step needs no new
# machinery, and the hyperbolic midpoint of an edge is the Minkowski-
# normalized sum of its endpoints on the hyperboloid.  For a face
# CENTRED at the origin the Klein disk preserves central angles and
# maps hyperbolic circumradius R to Klein radius tanh R, so one
# construction step obeys
#
#     tanh R' = tanh R * cos(ag)/cos(ag/2)      (star step, m >= 5)
#     tanh R' = tanh R * cos(ag/2)              (midpoint step)
#
# with ag = 2*pi/m -- the EUCLIDEAN ratio applied to tanh R.  (Proof by
# the hyperbolic right-triangle relation tanh(leg) = tanh(hyp) cos(angle):
# the perpendicular foot from the centre to the alternate-vertex chord
# has tanh d = tanh R cos ag, and the inner vertex sits on that chord
# at angle ag/2 from the foot, so tanh d = tanh R' cos(ag/2).)  As
# R -> 0, tanh R -> R and the ratio degenerates to the Euclidean
# cos(ag)/cos(ag/2) of `spidron_math.arm_ratio` -- checked in
# `_selftest`.  Triangles or squares admit no alternate-vertex polygon
# (the ratio is <= 0 for m < 5), so those faces take the midpoint
# (whirl) step instead, which is what Kaplan's {3,8} and {4,6} pictures
# show.  Colourings follow his gallery: alternating two-tone round each
# tile's arms, a checkerboard variant that flips the two-tone on
# alternating tiles, and concentric ring banding.
#
# Color is by side count, tile type, uniform, or PARITY.  Parity is
# the classic alternating two-tone: the faces are 2-colored so the
# color flips across every shared edge (a proper 2-coloring of the
# face-adjacency graph, seeded by reflection-word parity).  It
# alternates cleanly only when the tiling is BIPARTITE (q even --
# {5,4}, {6,4}, ...); for odd q ({3,7}, {8,3}, ...) the faces are not
# 2-colorable and the mode shows the unavoidable frustration seam
# (the snub, full of triangles, is likewise not 2-colorable).
#
# References:
# - W. A. Wythoff -- the Wythoff (kaleidoscopic) construction of uniform
#   tilings from a triangle reflection group.
# - H. S. M. Coxeter, "Regular Polytopes" (1948) and his work on
#   reflection groups and the {p,q} Schlafli symbol.
# - Henri Poincare -- the conformal disk model of the hyperbolic plane.
# - Henry Segerman, "Visualizing Mathematics with 3D Printing" (2016) --
#   hyperbolic surface models.
# - Craig S. Kaplan, "Hyperbolic Spidrons", 2007,
#   https://cs.uwaterloo.ca/~csk/other/spidron/ -- spidron nests drawn
#   inside the faces of regular hyperbolic tilings by iterating
#   inscribed regular and star polygons, with the alternating two-tone
#   and checkerboard colourings the Spidron Fill modes reproduce.
# - Daniel Erdely, "Some Surprising New Properties of the Spidrons",
#   Bridges 2005 Conference Proceedings, pp. 179-186 -- the original
#   (Euclidean) spidron figure the fill generalises.

bl_info = {
    "name": "Hyperbolic Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Hyperbolic Tiling",
    "description": "Uniform (Wythoff) hyperbolic tilings on (p,q,r) "
                   "triangle groups, as flat disk polygons or curved "
                   "3D surface models",
    "category": "Add Mesh",
}



from math import (acosh, atan2, atanh, cos, cosh, pi, sin, sinh, sqrt,
                  tanh)

import numpy as np

try:
    from .patterns import common as pc
    from . import tiling_generator as tg
    from . import spidron_math as sm
except Exception:                       # legacy single-file / CLI use
    from patterns import common as pc
    import tiling_generator as tg
    import spidron_math as sm
# The mathematics lives in the sibling `hyperbolic` engine package;
# this module is the Blender layer over it.
try:
    from .hyperbolic.tilings import (FORM_ITEMS, _kind_for, _slab_cell,
                                         build_surface_cells, build_uniform,
                                         tiling_faces, _parity_colors,
                                         _to_klein, _to_poincare,
                                         _normalize_h)
except ImportError:  # flat import outside the package
    from hyperbolic.tilings import (FORM_ITEMS, _kind_for, _slab_cell,
                                        build_surface_cells, build_uniform,
                                        tiling_faces, _parity_colors,
                                        _to_klein, _to_poincare,
                                        _normalize_h)








# --------------------------------------------------------------------
# Minkowski helpers
# --------------------------------------------------------------------



















# --------------------------------------------------------------------
# (p, q, r) triangle group
# --------------------------------------------------------------------









# --------------------------------------------------------------------
# Wythoff seed
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# Face assembly
# --------------------------------------------------------------------



































# --------------------------------------------------------------------
# Hyperbolic spidrons (Kaplan)
# --------------------------------------------------------------------
#
# The nest is built entirely in the Klein disk, where hyperbolic
# geodesics are straight chords: the star step ("connect alternate
# vertices") is plain 2D line intersection, and the midpoint step takes
# the hyperbolic midpoint -- the Minkowski-normalized sum of the edge's
# endpoints on the hyperboloid.  See the module header for the derived
# radius recursions and their Euclidean degeneration.

def _lift_klein(x, y):
    """Klein disk -> hyperboloid upper sheet: (x, y) -> t (x, y, 1)
    with t = 1/sqrt(1 - x^2 - y^2)."""
    t = 1.0 / sqrt(max(1.0 - x * x - y * y, 1e-15))
    return np.array([x * t, y * t, t])


def hyper_star_radius(R, m):
    """Hyperbolic circumradius of the alternate-vertex polygon
    inscribed in a regular m-gon of hyperbolic circumradius R:

        tanh R' = tanh R * cos(ag)/cos(ag/2),   ag = 2*pi/m.

    Derivation: drop a perpendicular from the centre to the chord
    V_(k-1) V_(k+1).  In the right triangle (centre, foot, V) the
    hyperbolic relation tanh(leg) = tanh(hyp) cos(angle) gives
    tanh d = tanh R cos(ag); the inner vertex lies on the same chord at
    central angle ag/2 from the foot, so tanh d = tanh R' cos(ag/2).
    As R -> 0 this degenerates to the Euclidean R'/R =
    cos(ag)/cos(ag/2) of `spidron_math.arm_ratio`.  Positive only for
    m >= 5, exactly as in the Euclidean case."""
    return atanh(tanh(R) * sm.arm_ratio(m))


def hyper_mid_radius(R, m):
    """Hyperbolic circumradius of the edge-midpoint polygon of a
    regular m-gon of circumradius R: tanh R' = tanh R * cos(pi/m)
    (one application of the same right-triangle relation).  Works for
    every m >= 3 -- the step used where the star step degenerates."""
    return atanh(tanh(R) * cos(pi / m))


def _isect(a, b, c, d):
    """Intersection of 2D lines ab and cd (Klein chords, i.e. the
    hyperbolic geodesics through those vertex pairs)."""
    r = b - a
    s = d - c
    den = r[0] * s[1] - r[1] * s[0]
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
    return a + t * r


def _nest_klein(K, rings):
    """The spidron nest inside one Klein-model polygon.

    K is the ordered (counterclockwise) list of the face's Klein
    corners.  Each ring inscribes the next polygon -- the star step
    (alternate-vertex chords) for m >= 5, the midpoint (whirl) step for
    triangles and squares -- and triangulates the annulus between them.
    Returns (tris, arm_ix, ring_ix, core): triangles as Klein 2D vertex
    triples, the spiral arm and ring of each, and the innermost polygon
    left open at the centre (as in Kaplan's pictures, where the white
    tile centres are the nests' unreached limits)."""
    m = len(K)
    cur = [np.asarray(v, float) for v in K]
    tris, arms, rixs = [], [], []
    for ri in range(rings):
        if m >= 5:
            # star step: inner vertex W_k between V_k and V_(k+1) is
            # the crossing of chords (V_(k-1), V_(k+1)) and
            # (V_k, V_(k+2)); the annulus is the 2m triangles of the
            # Euclidean kernel (`spidron_math.ring_triangles`)
            W = [_isect(cur[k - 1], cur[(k + 1) % m],
                        cur[k], cur[(k + 2) % m]) for k in range(m)]
            for k in range(m):
                k1 = (k + 1) % m
                tris.append((cur[k], cur[k1], W[k]))
                arms.append(k)
                rixs.append(ri)
                tris.append((W[k], cur[k1], W[k1]))
                arms.append(k)
                rixs.append(ri)
        else:
            # midpoint step: W_k is the HYPERBOLIC midpoint of edge
            # (V_k, V_(k+1)); the annulus is the m cut-off corner
            # triangles
            W = []
            for k in range(m):
                a, b = cur[k], cur[(k + 1) % m]
                h = _normalize_h(_lift_klein(a[0], a[1])
                                 + _lift_klein(b[0], b[1]))
                W.append(_to_klein(h))
            for k in range(m):
                k1 = (k + 1) % m
                tris.append((W[k], cur[k1], W[k1]))
                arms.append(k)
                rixs.append(ri)
        cur = W
    return tris, arms, rixs, cur


def _tri_poly(tri, model, density=10.0, cap=8):
    """One nest triangle as a flat-model polygon: its Klein corners for
    KLEIN (chords are the geodesics), or with each edge subdivided
    along the hyperboloid geodesic for POINCARE, where geodesics curve.
    Subdivision is adaptive -- long outer edges get up to `cap`
    samples, the deep tiny rings stay plain triangles."""
    if model == 'KLEIN':
        return np.array([(v[0], v[1]) for v in tri])
    pts = []
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        n = int(min(cap, max(1, np.ceil(np.hypot(b[0] - a[0],
                                                 b[1] - a[1]) * density))))
        ha = _lift_klein(a[0], a[1])
        hb = _lift_klein(b[0], b[1])
        for s in range(n):
            t = s / float(n)
            pts.append(_to_poincare(_normalize_h((1.0 - t) * ha + t * hb)))
    return np.array(pts)


SPIDRON_COLOR_ITEMS = [
    ('ARM', "Arms",
     "Alternating two-tone round each tile: alternate spiral arms take "
     "the two materials, Kaplan's per-tile colouring (a tile with an "
     "odd number of sides cannot alternate fully and keeps one seam)"),
    ('RING', "Rings",
     "Alternate the rings instead of the arms, banding each tile in "
     "concentric two-tone layers -- on square tiles this is the nested "
     "four-pointed-star look of Kaplan's pictures"),
    ('UNIFORM', "Uniform", "A single material"),
]


def build_spidron_fill(p, q, r, form, model, depth, max_tiles, rings,
                       color_mode='ARM', checker=False):
    """Spidron-fill every face of the tiling, for a flat disk model.
    Returns (polys, mats): one flat polygon per nest triangle and its
    material index.

    All tiles of the tiling are congruent, so all get the same nest --
    except that a tile's rings stop once its current polygon drops
    below rendering size (about a pixel of the disk), which caps the
    triangle count on the rim without leaving the visibly truncated
    open cores a fixed small ring count would.  The innermost polygon
    of every nest is left open, as in Kaplan's pictures.

    `checker` flips the two-tone's phase on alternating tiles (via the
    proper face 2-coloring; clean exactly when the tiling is bipartite,
    i.e. even q)."""
    faces = tiling_faces(p, q, r, form, depth, max_tiles)
    tilepar = (_parity_colors(faces) if checker else [0] * len(faces))
    polys, mats = [], []
    for i, (fp, _nsides, _pa) in enumerate(faces):
        K = [np.asarray(_to_klein(v)) for v in fp]
        mfc = len(K)
        span = max(np.hypot(K[(k + 1) % mfc][0] - K[k][0],
                            K[(k + 1) % mfc][1] - K[k][1])
                   for k in range(mfc))
        # rings until the nest is sub-pixel: span * ratio^k <= 0.006
        ratio = sm.arm_ratio(mfc) if mfc >= 5 else cos(pi / mfc)
        if span <= 0.006:
            rk = 1
        else:
            need = int(np.ceil(np.log(0.006 / span) / np.log(ratio)))
            rk = min(rings, max(1, need))
        tris, arms, rixs, _core = _nest_klein(K, rk)
        base = int(tilepar[i])
        for t, ai, ri in zip(tris, arms, rixs):
            polys.append(_tri_poly(t, model))
            if color_mode == 'RING':
                mats.append((ri + base) % 2)
            elif color_mode == 'UNIFORM':
                mats.append(0)
            else:                       # ARM two-tone
                mats.append((ai + base) % 2)
    return polys, mats


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------



try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


MODEL_ITEMS = [
    ('POINCARE', "Poincare Disk",
     "Conformal disk: geodesic edges curve (flat 2D)"),
    ('KLEIN', "Klein Disk",
     "Projective disk: geodesics are straight chords (flat 2D)"),
    ('HEMISPHERE', "Hemisphere",
     "Klein disk lifted onto the upper unit hemisphere (3D)"),
    ('PSEUDOSPHERE', "Pseudosphere",
     "Wrapped isometrically onto the tractricoid via the upper "
     "half plane (3D)"),
]

_FLAT = {'POINCARE', 'KLEIN'}


if _IN_BLENDER:

    class MESH_OT_hyperbolic_tiling_add(bpy.types.Operator,
                                        AddObjectHelper):
        """Add a hyperbolic tiling: any Wythoff form on a (p,q,r)
        triangle group, as a flat disk or a curved 3D surface"""
        bl_idname = "mesh.hyperbolic_tiling_add"
        bl_label = "Hyperbolic Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        form: EnumProperty(name="Form", items=FORM_ITEMS,
                           default='REGULAR',
                           description="Which Wythoff uniform form to "
                                       "build")
        p: IntProperty(
            name="p", default=7, min=3, max=12,
            description="Order of the first corner (p-gon symmetry)")
        q: IntProperty(
            name="q", default=3, min=3, max=12,
            description="Order of the second corner (q around a vertex)")
        r: IntProperty(
            name="r", default=2, min=2, max=12,
            description="Order of the third corner (2 = right triangle, "
                        "the classic regular families)")
        model: EnumProperty(name="Model", items=MODEL_ITEMS,
                            default='POINCARE',
                            description="Disk or curved 3D model the "
                                        "tiling is drawn in")
        depth: IntProperty(
            name="Depth", default=14, min=1, max=60,
            description="Reflection word length explored")
        max_tiles: IntProperty(
            name="Max Group Elements", default=8000, min=10,
            max=40000,
            description="Cap on triangle-group elements generated")
        spidron: BoolProperty(
            name="Spidron Fill", default=False,
            description="Fill every tile with its spidron nest: a "
                        "decreasing sequence of inscribed polygons "
                        "whose annuli triangulate into spiral arms -- "
                        "Kaplan's hyperbolic spidrons (flat disk "
                        "models)")
        spidron_rings: IntProperty(
            name="Spidron Rings", default=5, min=1, max=12,
            description="How many nest rings each tile gets; the "
                        "innermost polygon stays open. Distant tiny "
                        "tiles automatically take fewer rings")
        spidron_color: EnumProperty(
            name="Spidron Color", items=SPIDRON_COLOR_ITEMS,
            default='ARM',
            description="How the nest triangles are coloured -- "
                        "Kaplan's point that different colourings "
                        "bring out different geometric features")
        spidron_checker: BoolProperty(
            name="Checkerboard", default=False,
            description="Flip the two-tone's phase on alternating "
                        "tiles -- Kaplan's checkerboard variant. The "
                        "flip is clean when the tiling is bipartite "
                        "(even q); odd q leaves a frustration seam")
        color_by: EnumProperty(
            name="Color By",
            items=[('SIDES', "By Sides",
                    "Material per polygon side count (exact in Klein)"),
                   ('TYPE', "By Tile Type",
                    "Material per face side count / orbit"),
                   ('PARITY', "Parity (Alternating)",
                    "Classic two-tone by reflection-word parity; "
                    "alternates cleanly only for even q (bipartite), "
                    "shows a frustration seam for odd q"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='SIDES',
            description="How the tiles are colored")
        hide_off_parity: BoolProperty(
            name="Hide Off-Parity Tiles", default=False,
            description="In Parity color mode, omit the 'off' (second) "
                        "parity class so only the 'on' tiles render, "
                        "leaving a sparse alternating pattern")
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles (flat models)")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat tiling; > 0 extrudes each tile into a "
                        "raised plaque (flat models)")
        backing: BoolProperty(
            name="Backing Disk", default=False,
            description="Add a backing slab under the tiles, for a solid "
                        "disk plaque (flat models)")
        base: FloatProperty(
            name="Base Thickness", default=0.05, min=0.01, max=1.0,
            description="Backing slab thickness (flat models)")
        thickness: FloatProperty(
            name="Shell Thickness", default=0.0, min=0.0, max=0.5,
            description="0 = a single zero-thickness surface; > 0 makes "
                        "each tile a solid shell (curved models)")
        y_cap: FloatProperty(
            name="Cusp Cap", default=6.0, min=1.5, max=50.0,
            description="Clip the pseudosphere cusp at upper-half-plane "
                        "height y (the horn gets thin)")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object "
                        "(parented to an empty)")

        def execute(self, context):
            if (1.0 / self.p + 1.0 / self.q + 1.0 / self.r
                    >= 1.0 - 1e-9):
                self.report({'ERROR'},
                            "(%d,%d,%d) is not hyperbolic: need "
                            "1/p + 1/q + 1/r < 1" %
                            (self.p, self.q, self.r))
                return {'CANCELLED'}
            label = dict((k, v) for k, v, _ in FORM_ITEMS)[self.form]
            name = "Hyperbolic %s (%d,%d,%d)" % (
                label, self.p, self.q, self.r)
            try:
                if self.model in _FLAT and self.spidron:
                    name = "Hyperbolic Spidron %s (%d,%d,%d)" % (
                        label, self.p, self.q, self.r)
                    polys, mats = build_spidron_fill(
                        self.p, self.q, self.r, self.form, self.model,
                        self.depth, self.max_tiles,
                        int(self.spidron_rings), self.spidron_color,
                        self.spidron_checker)
                    if not polys:
                        self.report({'ERROR'}, "no tiles generated")
                        return {'CANCELLED'}
                    cells = tg.cells_from_polys(
                        lambda a, b: (polys, mats), 1, 1, 'TYPE',
                        self.margin, self.height, trim=False)
                    if self.backing and not self.separate:
                        cells.append(_slab_cell(1.0, self.base, 1))
                elif self.model in _FLAT:
                    polys, sides, parities = build_uniform(
                        self.p, self.q, self.r, self.form, self.model,
                        self.depth, self.max_tiles)
                    if not polys:
                        self.report({'ERROR'}, "no tiles generated")
                        return {'CANCELLED'}
                    if self.color_by == 'PARITY' and self.hide_off_parity:
                        keep = [i for i, pa in enumerate(parities)
                                if pa == 0]
                        polys = [polys[i] for i in keep]
                        sides = [sides[i] for i in keep]
                        parities = [parities[i] for i in keep]
                        if not polys:
                            self.report({'ERROR'},
                                        "no tiles left after hiding "
                                        "off-parity")
                            return {'CANCELLED'}
                    # Precompute the material index per tile from the
                    # TRUE side count and drive cells_from_polys via
                    # TYPE, so colors are correct even in Poincare
                    # (where edge subdivision inflates len(poly)).
                    mats = [_kind_for(s, pa, self.color_by)
                            for s, pa in zip(sides, parities)]
                    cells = tg.cells_from_polys(
                        lambda a, b: (polys, mats), 1, 1, 'TYPE',
                        self.margin, self.height, trim=False)
                    if self.backing and not self.separate:
                        cells.append(_slab_cell(1.0, self.base, 1))
                else:
                    if self.spidron:
                        self.report({'WARNING'},
                                    "Spidron fill applies to the flat "
                                    "disk models; built the plain "
                                    "surface")
                    cells = build_surface_cells(
                        self.p, self.q, self.r, self.form, self.model,
                        self.depth, self.max_tiles, self.color_by,
                        self.thickness, self.y_cap,
                        hide_off=(self.color_by == 'PARITY'
                                  and self.hide_off_parity))
            except ValueError as err:
                self.report({'ERROR'}, str(err))
                return {'CANCELLED'}
            if not cells:
                self.report({'ERROR'}, "no tiles survived clipping")
                return {'CANCELLED'}
            obj = pc.emit(context, name, cells, self.separate,
                          fit=True, operator=self)
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
            for prop in ('form', 'p', 'q', 'r', 'model', 'depth',
                         'max_tiles'):
                lay.prop(self, prop)
            spid = self.spidron and self.model in _FLAT
            if self.model in _FLAT:
                lay.prop(self, 'spidron')
            if spid:
                lay.prop(self, 'spidron_rings')
                lay.prop(self, 'spidron_color')
                if self.spidron_color != 'UNIFORM':
                    lay.prop(self, 'spidron_checker')
            else:
                lay.prop(self, 'color_by')
                if self.color_by == 'PARITY':
                    lay.prop(self, 'hide_off_parity')
            if self.model in _FLAT:
                lay.prop(self, 'margin')
                lay.prop(self, 'height')
                lay.prop(self, 'backing')
                if self.backing:
                    lay.prop(self, 'base')
            else:
                lay.prop(self, 'thickness')
                if self.model == 'PSEUDOSPHERE':
                    lay.prop(self, 'y_cap')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.hyperbolic_tiling_add",
                             icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_hyperbolic_tiling_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_hyperbolic_tiling_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _coverage_disk(polys, radius=0.72, samples=41, off=(0.0137, 0.0071)):
    """Point-coverage over an interior sub-disk (Klein).  Sample a grid,
    keep points inside `radius`, and count how many tiles cover each.
    Returns (tested, gaps, overlaps): points covered by 0, and by >= 2,
    tiles.  A per-tile bounding box prefilters the point-in-polygon
    tests so the sweep stays fast on dense patches."""
    boxes = [(p, p[:, 0].min(), p[:, 0].max(),
              p[:, 1].min(), p[:, 1].max()) for p in polys]
    r2 = radius * radius
    tested = gaps = overs = 0
    for ix in range(samples):
        x = -radius + 2.0 * radius * ix / (samples - 1) + off[0]
        for iy in range(samples):
            y = -radius + 2.0 * radius * iy / (samples - 1) + off[1]
            if x * x + y * y > r2:
                continue
            cnt = 0
            for p, x0, x1, y0, y1 in boxes:
                if x < x0 or x > x1 or y < y0 or y > y1:
                    continue
                if tg._pip(p, x, y):
                    cnt += 1
                    if cnt >= 2:
                        break
            tested += 1
            if cnt == 0:
                gaps += 1
            elif cnt >= 2:
                overs += 1
    return tested, gaps, overs


def _parity_defects(polys, parities, radius=0.9):
    """Count edge-adjacent face pairs (in an interior window) whose
    parity colors clash.  0 for a bipartite (even-q) tiling; positive
    where odd q forces a frustration seam.  Adjacency = two faces that
    share two nearly-coincident boundary vertices (an edge)."""
    import itertools
    cents = [p.mean(axis=0) for p in polys]
    keep = [i for i, c in enumerate(cents)
            if c[0] * c[0] + c[1] * c[1] < radius * radius]
    vsets = {i: {(round(x, 3), round(y, 3)) for x, y in polys[i]}
             for i in keep}
    clash = adj = 0
    for i, j in itertools.combinations(keep, 2):
        if len(vsets[i] & vsets[j]) >= 2:            # share an edge
            adj += 1
            if parities[i] == parities[j]:
                clash += 1
    return adj, clash


def _selftest():
    CASES = [(7, 3, 2), (5, 4, 2), (8, 3, 2), (4, 3, 3)]
    FORMS = ['REGULAR', 'TRUNCATED', 'RECTIFIED', 'BITRUNCATED',
             'CANTELLATED', 'OMNITRUNCATED', 'SNUB']
    DEPTH, MAXT = 48, 1200
    all_ok = True
    passed_forms = set()
    for form in FORMS:
        form_ok = True
        for (p, q, r) in CASES:
            polys, sides, parities = build_uniform(p, q, r, form,
                                                   'KLEIN', DEPTH, MAXT)
            tested, gaps, overs = _coverage_disk(polys)
            ok = (tested > 0 and gaps == 0 and overs == 0
                  and len(polys) > 0)
            form_ok = form_ok and ok
            print("%-14s {%d,%d,%d}  tiles=%5d sides=%-14s "
                  "sampled=%4d gaps=%3d overlaps=%3d  %s"
                  % (form, p, q, r, len(polys), str(sorted(set(sides))),
                     tested, gaps, overs, "OK" if ok else "BAD"))
        if form_ok:
            passed_forms.add(form)
        all_ok = all_ok and form_ok
    print("forms verified:",
          ", ".join(f for f in FORMS if f in passed_forms))

    # every output model builds a non-empty valid mesh for a couple forms
    print("--- models ---")
    models_ok = True
    for model in ('POINCARE', 'KLEIN', 'HEMISPHERE', 'PSEUDOSPHERE'):
        for form in ('REGULAR', 'TRUNCATED'):
            if model in ('POINCARE', 'KLEIN'):
                polys, _s, _pa = build_uniform(7, 3, 2, form, model,
                                               16, 1500)
                nf = len(polys)
                nv = sum(len(p) for p in polys)
            else:
                cells = build_surface_cells(7, 3, 2, form, model, 16,
                                            1500)
                nf = sum(len(cf) for _cv, cf, _cm in cells)
                nv = sum(len(cv) for cv, _cf, _cm in cells)
            good = nf > 0 and nv > 0
            models_ok = models_ok and good
            print("%-13s %-10s  cells/polys ok  V~%-6d F~%-6d  %s"
                  % (model, form, nv, nf, "OK" if good else "BAD"))
    all_ok = all_ok and models_ok

    # PARITY 2-coloring: clean checkerboard for even q, seam for odd q
    print("--- parity ---")
    parity_ok = True
    for (p, q, r), expect in [((5, 4, 2), 'clean'), ((6, 4, 2), 'clean'),
                              ((7, 3, 2), 'seam'), ((8, 3, 2), 'seam')]:
        polys, sides, parities = build_uniform(p, q, r, 'REGULAR',
                                               'KLEIN', DEPTH, MAXT)
        adj, clash = _parity_defects(polys, parities)
        want_clean = (expect == 'clean')
        good = adj > 0 and ((clash == 0) == want_clean)
        parity_ok = parity_ok and good
        print("REGULAR {%d,%d,%d} q%s  edges=%3d clashes=%3d expect=%-5s  %s"
              % (p, q, r, "even" if q % 2 == 0 else "odd ", adj, clash,
                 expect, "OK" if good else "BAD"))
    all_ok = all_ok and parity_ok

    # By-Sides coloring must survive edge subdivision: a mixed form
    # (14-gons + triangles) must yield >1 distinct material index in
    # BOTH flat models, not collapse to one.
    print("--- color ---")
    color_ok = True
    for model in ('KLEIN', 'POINCARE'):
        polys, sides, parities = build_uniform(7, 3, 2, 'TRUNCATED',
                                               model, 16, 1500)
        mats = [_kind_for(s, pa, 'SIDES') for s, pa in zip(sides,
                                                           parities)]
        ndistinct = len(set(mats))
        good = ndistinct > 1
        color_ok = color_ok and good
        print("TRUNCATED {7,3,2} %-9s SIDES  distinct materials=%d  %s"
              % (model, ndistinct, "OK" if good else "BAD"))
    all_ok = all_ok and color_ok

    # Spidron fill (Kaplan).  The closed-form radius recursions are
    # checked against a fully independent numeric construction on the
    # hyperboloid, their Euclidean degeneration is pinned with a
    # shrinking-residual test, and the nest is verified to partition
    # every tile exactly.
    print("--- spidron ---")
    spid_ok = True

    def sh_area(P):
        a = 0.0
        for i in range(len(P)):
            x0, y0 = P[i]
            x1, y1 = P[(i + 1) % len(P)]
            a += x0 * y1 - x1 * y0
        return 0.5 * a

    # closed form vs the chord construction itself: build a centred
    # regular m-gon of hyperbolic circumradius R on the hyperboloid,
    # intersect the alternate-vertex chords in the Klein disk, and
    # measure the inner vertex's true hyperbolic distance from the
    # centre (acosh of its hyperboloid t) and its central angle.
    worst_r = worst_a = 0.0
    for m in (5, 6, 7, 8):
        for R in (0.25, 0.8, 1.6):
            V = [np.array([sinh(R) * cos(2 * pi * k / m),
                           sinh(R) * sin(2 * pi * k / m), cosh(R)])
                 for k in range(m)]
            K = [np.asarray(_to_klein(v)) for v in V]
            W0 = _isect(K[-1], K[1], K[0], K[2])
            d = acosh(_lift_klein(W0[0], W0[1])[2])
            worst_r = max(worst_r, abs(d - hyper_star_radius(R, m)))
            worst_a = max(worst_a, abs(atan2(W0[1], W0[0]) - pi / m))
    good = worst_r < 1e-9 and worst_a < 1e-9
    spid_ok = spid_ok and good
    print("star step: tanh R' = tanh R cos(ag)/cos(ag/2)  "
          "radius err=%.1e  turn err=%.1e  %s"
          % (worst_r, worst_a, "OK" if good else "BAD"))
    worst_m = 0.0
    for m in (3, 4, 6):
        for R in (0.4, 1.2):
            V0 = np.array([sinh(R), 0.0, cosh(R)])
            V1 = np.array([sinh(R) * cos(2 * pi / m),
                           sinh(R) * sin(2 * pi / m), cosh(R)])
            d = acosh(_normalize_h(V0 + V1)[2])
            worst_m = max(worst_m, abs(d - hyper_mid_radius(R, m)))
    good = worst_m < 1e-9
    spid_ok = spid_ok and good
    print("midpoint step: tanh R' = tanh R cos(pi/m)      "
          "radius err=%.1e  %s" % (worst_m, "OK" if good else "BAD"))

    # Euclidean degeneration: R'/R -> arm_ratio(m) as R -> 0, with the
    # residual shrinking quadratically (refine and confirm it SHRINKS,
    # rather than trusting one endpoint sample)
    for m in (5, 7):
        e1 = abs(hyper_star_radius(1e-2, m) / 1e-2 - sm.arm_ratio(m))
        e2 = abs(hyper_star_radius(1e-3, m) / 1e-3 - sm.arm_ratio(m))
        below = all(hyper_star_radius(R, m) < sm.arm_ratio(m) * R
                    for R in (0.2, 0.7, 1.5))
        good = e1 < 1e-3 and e2 < 0.05 * e1 and below
        spid_ok = spid_ok and good
        print("m=%d degenerates to Euclidean ratio %.6f  "
              "res %.1e -> %.1e  hyperbolic < Euclidean %s  %s"
              % (m, sm.arm_ratio(m), e1, e2, below,
                 "OK" if good else "BAD"))

    # on the real tilings: the central face has the textbook {p,q}
    # circumradius cosh R = cot(pi/p) cot(pi/q) (the right triangle
    # centre / edge-midpoint / vertex has angles pi/p and pi/q, and a
    # hyperbolic right triangle obeys cosh(hyp) = cot A cot B), and
    # its nest
    # partitions the tile exactly -- triangle areas sum to face area
    # minus the open core (Klein areas: same point sets, so a valid
    # partition test), every triangle positively oriented.
    for (p_, q_) in ((7, 3), (5, 4), (4, 6), (3, 8), (8, 3)):
        faces = tiling_faces(p_, q_, 2, 'REGULAR', 24, 400)
        cents = [np.mean([_to_klein(v) for v in fp], axis=0)
                 for fp, _n, _pa in faces]
        ci = min(range(len(faces)),
                 key=lambda i: float(cents[i] @ cents[i]))
        fp = faces[ci][0]
        R_face = acosh(float(np.mean([v[2] for v in fp])))
        want = acosh((cos(pi / p_) / sin(pi / p_))
                     * (cos(pi / q_) / sin(pi / q_)))
        K = [np.asarray(_to_klein(v)) for v in fp]
        tris, arms, rixs, core = _nest_klein(K, 5)
        per_ring = 2 * p_ if p_ >= 5 else p_
        tri_area = sum(sh_area(t) for t in tris)
        part_err = abs(tri_area - (sh_area(K) - sh_area(core)))
        good = (abs(R_face - want) < 1e-6
                and len(tris) == 5 * per_ring
                and all(sh_area(t) > 0.0 for t in tris)
                and part_err < 1e-12
                and sorted(set(arms)) == list(range(p_))
                and max(rixs) == 4)
        spid_ok = spid_ok and good
        print("{%d,%d} R=%.4f (want %.4f)  %d tris partition the "
              "tile, err=%.1e  %s"
              % (p_, q_, R_face, want, len(tris), part_err,
                 "OK" if good else "BAD"))

    # the full fill builds for both flat models, two-tone material
    # indices only, everything inside the unit disk
    for (p_, q_), mode, chk_, model in (
            ((8, 3), 'ARM', False, 'POINCARE'),
            ((4, 6), 'RING', True, 'POINCARE'),
            ((6, 4), 'ARM', True, 'KLEIN')):
        polys, mats = build_spidron_fill(p_, q_, 2, 'REGULAR', model,
                                         20, 400, 4, mode, chk_)
        rmax = max(float(np.hypot(v[0], v[1]))
                   for pp in polys for v in pp)
        good = (len(polys) > 200 and set(mats) <= {0, 1}
                and all(len(pp) >= 3 for pp in polys)
                and rmax <= 1.0 + 1e-9)
        spid_ok = spid_ok and good
        print("fill {%d,%d} %-4s chk=%d %-8s  %5d tris  mats=%s  "
              "rmax=%.4f  %s"
              % (p_, q_, mode, chk_, model, len(polys),
                 sorted(set(mats)), rmax, "OK" if good else "BAD"))
    # the checkerboard really flips the phase tile to tile on a
    # bipartite case: {4,6} with and without it must differ on some
    # triangles but not all
    pa_, ma_ = build_spidron_fill(4, 6, 2, 'REGULAR', 'KLEIN',
                                  20, 400, 3, 'ARM', False)
    pc_, mc_ = build_spidron_fill(4, 6, 2, 'REGULAR', 'KLEIN',
                                  20, 400, 3, 'ARM', True)
    diff = sum(1 for a, b in zip(ma_, mc_) if a != b)
    good = len(ma_) == len(mc_) and 0 < diff < len(ma_)
    spid_ok = spid_ok and good
    print("checkerboard flips the two-tone on %d/%d triangles  %s"
          % (diff, len(ma_), "OK" if good else "BAD"))
    all_ok = all_ok and spid_ok

    print("RESULT:", "OK" if all_ok else "BAD")
    assert all_ok
