
# Ruled Surfaces generator for Blender: the family of surfaces swept
# by a straight line (a "ruling") moving through space.  Every mode
# here is built from the general ruled-surface recipe
#
#       S(u, v) = b(u) + v * d(u)
#
# a directrix (base curve) b(u) plus v times a ruling direction d(u),
# or from a right-conoid variant S(u,v) = (v cos u, v sin u, h(u)),
# or from a bilinear patch spanning four skew points.  Because the
# generator is always a straight segment, each surface can optionally
# be rendered as its literal RULINGS -- thin rods or bare curves --
# optionally alongside its directrix / boundary rails (the two circles,
# the two torus knots, ...), reproducing the look of string / stick
# sculptures (George Hart's stick hyperboloids, hyperbolic-paraboloid
# string art).
#
# The two doubly-ruled modes, the hyperboloid and the hyperbolic
# paraboloid, carry a second family of rulings crossing the first -- and
# so do the spiral ruled surface and the concentric toroidal knots, whose
# second families are laid on their surfaces as curves (see SPIRAL and
# KNOT_SPAN below).  All four can also be
# rendered as WOVEN RIBBONS: every strand of both families a flat ribbon
# lying in the surface, passing alternately over and under the ribbons of
# the other family (a plain weave, or a twill).  The weaving itself knows
# nothing about the surface and lives in `weaving/rulings.py`; a mode
# joins it by supplying its two families (see `_weave_input`).  The ribbons can also run on past
# the edge (Overhang): a ruling of these surfaces still lies on the
# surface when extended, so the lengthened ribbons keep crossing and the
# weave itself carries on beyond the rails.
#
# Every other straight-ruled mode weaves too, the way a basket does: its
# rulings are the stakes, and the weavers are the curves running along
# the surface a fixed way along the rulings -- copies of the base curve
# (`weft_weave`).  They cross every ruling exactly, and close into loops
# on a closed surface; a Mobius band's weavers go round twice before they
# meet themselves, since one lap brings each out on the opposite side of
# the band.  Where rulings run together -- a cone's apex, a conoid's
# axis, the seams where Guimard's surface and the milk carton pinch
# shut -- the ribbons stop short (Weave Gap): no weave can pass many
# strands over and under one another at a single point.  Only the
# compound helical cone, which has no rulings, has nothing to weave.
#
# Rods output can also be made solid: Separate Touching Rods bends any
# two rods that would pass through each other just far enough apart to
# clear, keeping every rod end on its rail (`separate_rods`, in the same
# engine module).
#
# Surface output takes a wall thickness one of two ways.  Solidify
# offsets the surface to both sides along its normals, which stays clean
# only while half the thickness is under the surface's reach: where the
# surface bends more tightly than that, pinches to a point, or passes
# through itself, the offset folds over into fins.  The concentric
# toroidal knots can do all three -- an inner (1, 3) knot and an outer
# (1, 6) knot at twice its scale even touch, at the six points where
# cos 3t = 1/4, and there a twist past about 24 degrees pinches the
# surface.  Fused Solid instead keeps every point within half the
# thickness of the surface: Poisson-disk samples of it are splatted into
# a sparse volume and meshed back, which gives one closed solid whatever
# the surface does, at the price of rounded rims and a heavier mesh
# (`fused_solid_params` sizes it).  The default, Automatic, builds the
# Solidify wall, looks for faces of it passing through one another
# (`count_crossings`), and falls back to Fused Solid only if it finds
# any.
#
# Modes
#   HYPERBOLOID   -- hyperboloid of one sheet from straight rulings
#     strung between two coaxial circles, the top circle rotated by a
#     twist angle.  The waist radius a = R cos(twist/2) is set purely
#     by the twist: 0 -> cylinder, 180 deg -> double cone.  Doubly
#     ruled: left- and right-handed families of rulings, drawn singly,
#     crossing, or woven.
#   HELICAL_CONE  -- the compound helical cone / Solomonic column: a
#     cone that is spirally fluted (N flutes winding helically) and
#     optionally wound a second time along a planetary helix, its
#     flutes fading from ornament at the base to arrises at the apex.
#   SPIRAL        -- Farris' spiral ruled surfaces: the hyperboloid's
#     generating circle replaced by a logarithmic spiral (or an
#     n-fold-symmetric rosette), swept with a tangent+vertical ruling.
#     A log spiral is carried onto itself by turning and scaling, so its
#     left family is laid on the surface the way the knot span's is:
#     each strand winds back through the angle its right partner winds
#     forward (`left_rulings_spiral`), which on a circle is exactly the
#     hyperboloid's straight left ruling.
#   CONOID        -- right conoids S=(v cos u, v sin u, h(u)):
#     Plucker's conoid / cylindroid (h = c sin 2u), the n-fold
#     generalization, the Wallis conical edge, Zindler's conoid
#     (h = a tan 2u, the cubic ruled surface z(x^2-y^2) = 2axy) and
#     the Whitney umbrella (a pinch-point ruled surface); plus the
#     parabolic conoid (parabola directrix, a^2 z = x(b^2 - y^2)) and
#     two ruled cones -- the sinusoidal cone z = k rho cos(n theta)
#     and the helicoidal cone (apex joined to a circular helix).
#   TANGENT_DEV   -- the tangent developable of a circular helix,
#     T(u,v) = c(u) + v c'(u): a flat-unrollable (developable) surface
#     whose edge of regression is the helix itself.
#   HELICOID      -- the right / oblique helicoid, a radial ruling
#     screwing up the axis; the right helicoid (slope 0) is minimal.
#   TWIST_STRIP   -- an n-half-twist ruled band; odd n is a Mobius
#     band (one-sided), even n an orientable twisted annulus.
#   NAMED         -- five ruled surfaces that are named after what they
#     were built for rather than after their equation: Gaudi's
#     sinusoidal conoid (the Sagrada Familia escoles roof), Guimard's
#     Art Nouveau surface, the milk carton / berlingot, the skew ruled
#     cubic (a conic and a line joined by a homography), and Monge's
#     surface of constant slope, the shape a heap of dry sand takes.
#   HYPAR         -- the doubly-ruled hyperbolic paraboloid, either as
#     z = c((x/a)^2 - (y/b)^2) or as the bilinear patch spanning four
#     user-set skew corner points (the surface of any four points in
#     general position).  Its two ruling families can likewise be drawn
#     singly, crossing, or woven.
#   KNOT_SPAN     -- a ruled surface strung between two concentric
#     (p, q) torus knots: straight rulings interpolate S = inner(u)(1-v)
#     + outer(u + twist) v between an inner and an outer toroidal knot
#     sampled on a shared parameter.  Those straight rulings are the
#     right family.  The left family is laid on the SAME surface, so the
#     two fill it in together and cross the way a stick hyperboloid's
#     do: each left strand winds back about the axis by the angle its
#     right partner winds forward.  With two coaxial circles the surface
#     is a hyperboloid and the left strands are its straight left
#     rulings; on knotted rails no surface but a quadric carries two
#     straight families, so there the left strands are gently curved.
#     The outer curve degenerates to a plain circle (wound p times) when
#     its q = 0.  A straight-ruled cousin of the soap-film "knot to
#     knot" span in the minimal-surface toolkit.
#
# Every builder is pure python + numpy and runs without bpy, so this
# file self-tests standalone.  Seams where a parameter wraps are
# welded by index; apex/pinch collapses are handled per mode.
#
# References:
# - G. W. Hart, "Curved, yet Straight: Stick Hyperboloids," Bridges
#   2023 Conference Proceedings, pp. 251-258.
# - E. Jannasch & J. Macnab, "The Compound Helical Cone as Kinematic
#   Trace," Bridges 2023 Conference Proceedings, pp. 15-22.
# - F. A. Farris, "Spiral Ruled Surfaces," Bridges 2022 Conference
#   Proceedings, pp. 289-292; and F. A. Farris, "Creating
#   Symmetry" (Princeton Univ. Press, 2015).
# - Antoni Gaudi (1852-1926), the ruled roof of the Sagrada Familia
#   escoles; Hector Guimard (1867-1942).  Both parametrizations, and the
#   milk carton's, from R. Ferreol, "Encyclopedie des formes
#   mathematiques remarquables" (mathcurve.com), chapters "surface de
#   Gaudi", "surface de Guimard" and "berlingot".
# - Milk carton: H. M. Cundy & A. P. Rollett, "Mathematical Models"
#   (1951), 185-188.
# - Surface of constant slope: Gaspard Monge (1807); see also R. Iss
#   (1985).
# - Ruled cubics as a conic and a line joined by a homography: the
#   classification in R. Ferreol, ibid., chapter "cubique reglee".
# - J. Plucker, "On a New Geometry of Space" (1865); H. Whitney,
#   "The general type of singularity of a set of 2n-1 smooth
#   functions of n variables" (1943).
# - K. Zindler (1866-1934), the conoid z = a tan 2 theta; see
#   R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#   mathcurve.com, chapter "conoide de Zindler", for the cartesian
#   form and the tangentoid-crown directrix used here.  A converted
#   copy of the encyclopedia is in research/books/
#   mathcurve_encyclopedie_formes_mathematiques/.
# - Parabolic conoid (a2 z = x(b2 - y2), the roof-shell conoid with a
#   parabola directrix), sinusoidal cone (cylindrical equation
#   z = k rho cos(n theta), the cone over the sine-wave crown) and
#   helicoidal cone (x = a u cos v, y = a u sin v, z = b u v, the cone
#   joining an apex to a circular helix): R. Ferreol, ibid., chapters
#   "conoide parabolique", "cone sinusoidal" and "cone helicoidal".
# - S. A. Coons, "Surfaces for Computer-Aided Design of Space Forms,"
#   MIT Project MAC TR-41 (1967) -- the bilinear patch.
# - Torus knots (p, q): classical; see e.g. C. C. Adams, "The Knot
#   Book" (1994).  The knot-to-knot span is adapted here as a pure
#   ruled surface (cf. the soap-film version in the minimal-surface
#   toolkit).
# - D. Hilbert and S. Cohn-Vossen, "Anschauliche Geometrie" (1932),
#   chapter 1 -- the doubly ruled quadrics, whose two rulings through
#   each point are what the woven-ribbon output interlaces.
# - B. Grunbaum and G. C. Shephard, "Satins and Twills: An Introduction
#   to the Geometry of Fabrics," Mathematics Magazine 53 (1980),
#   139-161 -- plain weave and twills as over/under patterns.
# - H. Federer, "Curvature Measures," Transactions of the American
#   Mathematical Society 93 (1959), 418-491 -- the reach of a set: an
#   offset thinner than it cannot fold, one thicker can.
# - K. Museth, "VDB: High-Resolution Sparse Volumes with Dynamic
#   Topology," ACM Transactions on Graphics 32(3) (2013) -- the sparse
#   volume the Fused Solid wall is splatted into and meshed from.
# - Classical background: M. do Carmo, "Differential Geometry of
#   Curves and Surfaces" (1976); A. Gray, "Modern Differential
#   Geometry of Curves and Surfaces" (1997); D. Struik, "Lectures
#   on Classical Differential Geometry" (1950).

bl_info = {
    "name": "Ruled Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Hyperboloids, compound helical cones, "
                   "spiral ruled surfaces, conoids, tangent "
                   "developables, helicoids, twisted strips, "
                   "doubly-ruled hypars and concentric torus-knot "
                   "spans -- straight-line-swept surfaces, optionally "
                   "rendered as rulings or woven ribbons",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty, FloatVectorProperty,
                           StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:                                  # inside the math_art package
    from .weaving.rulings import (weave_rulings, crossing_clearance,
                                  extend_families, polyline_keep,
                                  segment_distance, separate_rods,
                                  plan_weave)
except ImportError:                   # flat import (test runner)
    from weaving.rulings import (weave_rulings, crossing_clearance,
                                 extend_families, polyline_keep,
                                 segment_distance, separate_rods,
                                 plan_weave)

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------
# shared mesh helpers
# --------------------------------------------------------------------

def _mesh_grid(P, wrap_u=False, wrap_v=False):
    """Quad-mesh a grid of points P (shape (nu, nv, 3)); wrap_u / wrap_v
    weld the last row/column back onto the first by index."""
    P = np.asarray(P, dtype=float)
    nu, nv = P.shape[0], P.shape[1]
    verts = [tuple(P[i, j]) for i in range(nu) for j in range(nv)]
    faces = []
    iu = nu if wrap_u else nu - 1
    jv = nv if wrap_v else nv - 1
    for i in range(iu):
        i1 = (i + 1) % nu
        for j in range(jv):
            j1 = (j + 1) % nv
            faces.append([i * nv + j, i1 * nv + j,
                          i1 * nv + j1, i * nv + j1])
    return verts, faces


def _capped(rings, apex):
    """Quad-mesh a stack of equal-length rings (each a list of points,
    the ring wrapping in its own direction) and fan the last ring to a
    single apex point.  Used by the helical cone."""
    m = len(rings[0])
    verts = [tuple(p) for ring in rings for p in ring]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(m):
            j1 = (j + 1) % m
            faces.append([i * m + j, (i + 1) * m + j,
                          (i + 1) * m + j1, i * m + j1])
    ai = len(verts)
    verts.append(tuple(apex))
    last = (len(rings) - 1) * m
    for j in range(m):
        faces.append([last + j, last + (j + 1) % m, ai])
    return verts, faces


def _edges(segments):
    """Build a bare wireframe (verts + edges, no faces) from the ruling
    segments (p0, p1): each ruling becomes one straight edge."""
    verts, edges = [], []
    for p0, p1 in segments:
        i = len(verts)
        verts.append(tuple(p0))
        verts.append(tuple(p1))
        edges.append((i, i + 1))
    return verts, edges


def _rods(segments, radius=0.02, sides=8):
    """Build a cylinder (rod) of the given radius along every segment
    (p0, p1); returns (verts, faces).  Rulings become physical sticks."""
    verts, faces = [], []
    for p0, p1 in segments:
        a = np.asarray(p0, dtype=float)
        b = np.asarray(p1, dtype=float)
        axis = b - a
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            continue
        w = axis / length
        ref = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 \
            else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(w, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(w, e1)
        base = len(verts)
        for k in range(sides):
            ang = _TWO_PI * k / sides
            off = radius * (math.cos(ang) * e1 + math.sin(ang) * e2)
            verts.append(tuple(a + off))
            verts.append(tuple(b + off))
        for k in range(sides):
            k1 = (k + 1) % sides
            faces.append([base + 2 * k, base + 2 * k1,
                          base + 2 * k1 + 1, base + 2 * k + 1])
        faces.append([base + 2 * k for k in range(sides)][::-1])
        faces.append([base + 2 * k + 1 for k in range(sides)])
    return verts, faces


def _tube(pts, closed=True, radius=0.02, sides=8):
    """Sweep one continuous tube along a polyline.

    `_rods` gives every segment its own capped cylinder, which is right
    for the rulings -- they really are separate sticks -- but wrong for
    a boundary rail, where consecutive chords share an endpoint.  Two
    capped cylinders meeting at an angle leave a visible lump at every
    joint, and on a rail that approximates a circle the result is a ring
    of bumps rather than a smooth hoop.

    Here the cross-sections are shared: one ring of `sides` points per
    polyline vertex, oriented on the average of the incoming and
    outgoing tangents, quad-stripped to its neighbours.  The frame is
    carried along by parallel transport so the tube does not spin about
    its own axis, and for a closed rail the leftover holonomy is spread
    evenly around the loop so the seam closes without a twist.
    """
    P = np.asarray(pts, dtype=float)
    if len(P) > 1 and closed and np.linalg.norm(P[0] - P[-1]) < 1e-12:
        P = P[:-1]                      # drop a duplicated closing point
    n = len(P)
    if n < 2:
        return [], []

    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T = np.empty_like(P)
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    L = np.linalg.norm(T, axis=1, keepdims=True)
    L[L < 1e-12] = 1.0
    T = T / L

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(T[0] @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    N = np.cross(T[0], ref)
    N /= np.linalg.norm(N)
    normals = [N]
    for i in range(1, n):
        prev, cur = T[i - 1], T[i]
        v = np.cross(prev, cur)
        s = float(np.linalg.norm(v))
        Nn = normals[-1]
        if s > 1e-12:                   # rotate the frame onto the new tangent
            axis = v / s
            ang = math.atan2(s, float(prev @ cur))
            c, sn = math.cos(ang), math.sin(ang)
            Nn = (Nn * c + np.cross(axis, Nn) * sn
                  + axis * float(axis @ Nn) * (1.0 - c))
        Nn = Nn - float(Nn @ cur) * cur
        ln = float(np.linalg.norm(Nn))
        Nn = normals[-1] if ln < 1e-12 else Nn / ln
        normals.append(Nn)

    if closed:
        # close the frame: whatever angle the transported normal has
        # drifted by after one lap gets unwound evenly along the loop
        last = normals[-1]
        v = np.cross(last, T[-1])
        drift = math.atan2(float(np.cross(last, normals[0]) @ T[0]),
                           float(last @ normals[0]))
        for i in range(n):
            a = -drift * i / n
            c, sn = math.cos(a), math.sin(a)
            Ni, Ti = normals[i], T[i]
            normals[i] = (Ni * c + np.cross(Ti, Ni) * sn
                          + Ti * float(Ti @ Ni) * (1.0 - c))

    verts, faces = [], []
    for i in range(n):
        Ti = T[i]
        Ni = normals[i]
        Bi = np.cross(Ti, Ni)
        for k in range(sides):
            ang = _TWO_PI * k / sides
            verts.append(tuple(P[i] + radius * (math.cos(ang) * Ni
                                                + math.sin(ang) * Bi)))
    rings = n if closed else n - 1
    for i in range(rings):
        a0 = i * sides
        a1 = ((i + 1) % n) * sides
        for k in range(sides):
            k1 = (k + 1) % sides
            faces.append([a0 + k, a0 + k1, a1 + k1, a1 + k])
    if not closed:
        faces.append(list(range(sides))[::-1])
        base = (n - 1) * sides
        faces.append([base + k for k in range(sides)])
    return verts, faces


def _tubes(loops, radius=0.02, sides=8):
    """`_tube` over a list of (points, closed) polylines, merged."""
    verts, faces = [], []
    for pts, closed in loops:
        v, f = _tube(pts, closed, radius, sides)
        o = len(verts)
        verts.extend(v)
        faces.extend([[i + o for i in q] for q in f])
    return verts, faces


def _simplify_polyline(P, tol):
    """The points of polyline P needed to stay within `tol` of it
    (`polyline_keep`, Ramer-Douglas-Peucker).  A curved ruling sampled
    densely enough to be smooth needs far fewer points to be swept as a
    tube, and one that came out straight becomes a stick."""
    P = np.asarray(P, dtype=float)
    if len(P) <= 2:
        return P
    return P[polyline_keep(P, tol)]


# --------------------------------------------------------------------
# 1. stick hyperboloid
# --------------------------------------------------------------------

def build_hyperboloid(radius=1.0, height=1.0, twist=120.0,
                      res_u=96, res_v=16):
    """Hyperboloid of one sheet as straight rulings between a bottom
    circle (radius R, z = -H) and a top circle rotated by `twist`
    degrees (z = +H).  The waist radius is a = R cos(twist/2)."""
    tw = math.radians(twist)
    u = np.linspace(0.0, _TWO_PI, res_u, endpoint=False)
    v = np.linspace(0.0, 1.0, res_v + 1)
    bx, by = radius * np.cos(u), radius * np.sin(u)
    tx, ty = radius * np.cos(u + tw), radius * np.sin(u + tw)
    P = np.empty((res_u, res_v + 1, 3))
    for j, vv in enumerate(v):
        P[:, j, 0] = bx * (1.0 - vv) + tx * vv
        P[:, j, 1] = by * (1.0 - vv) + ty * vv
        P[:, j, 2] = -height * (1.0 - vv) + height * vv
    return _mesh_grid(P, wrap_u=True)


def rulings_hyperboloid(radius=1.0, height=1.0, twist=120.0,
                        family='BOTH', n=48):
    """Ruling segments of the stick hyperboloid.  RIGHT twists the top
    end +twist, LEFT twists it -twist; BOTH gives the crossing string
    sculpture (the doubly-ruled surface)."""
    tw = math.radians(twist)
    segs = []
    for i in range(n):
        a = _TWO_PI * i / n
        b = (radius * math.cos(a), radius * math.sin(a), -height)
        if family in ('RIGHT', 'BOTH'):
            segs.append((b, (radius * math.cos(a + tw),
                             radius * math.sin(a + tw), height)))
        if family in ('LEFT', 'BOTH'):
            segs.append((b, (radius * math.cos(a - tw),
                             radius * math.sin(a - tw), height)))
    return segs


# --------------------------------------------------------------------
# 1b. concentric torus-knot span
# --------------------------------------------------------------------

def _torus_knot_at(p, q, t, scale=1.0, tube=1.0, major=2.0):
    """A (p, q) torus knot at parameters t (an array of any shape):
        r     = tube cos(q t) + major
        (x,y) = r (cos(p t), sin(p t)),   z = -tube sin(q t)
    all times `scale`.  q = 0 degenerates to a circle of radius
    (tube + major) wound p times."""
    t = np.asarray(t, dtype=float)
    r = np.cos(q * t) * tube + major
    return np.stack([r * np.cos(p * t), r * np.sin(p * t),
                     -np.sin(q * t) * tube], axis=-1) * scale


def _knot_curve(p, q, m, scale=1.0, tube=1.0, major=2.0, phase=0.0):
    """`_torus_knot_at` sampled at m points on t in [0, 2pi), endpoint
    excluded so the loop welds cleanly under wrap_u.  `phase` shifts
    every sample along the knot (t -> t + phase) without changing the
    curve itself."""
    return _torus_knot_at(p, q, np.linspace(0.0, _TWO_PI, m, endpoint=False)
                          + phase, scale, tube, major)


def _knot_span_at(t_inner, t_outer, p=2, q=3, knot_scale=1.0, tube=1.0,
                  inner_height=1.0, inner_lift=0.0, inner_rotation=0.0,
                  outer_p=0, outer_q=5, outer_scale=2.0, outer_tube=1.0,
                  outer_height=1.0, circle_radius=4.5):
    """The knot span's inner curve at parameters t_inner and outer curve
    at t_outer (arrays of any shape; points on the last axis).

    The inner curve is always a (p, q) torus knot, height-scaled, lifted
    and optionally rotated about z.  The outer curve is a second
    (outer_p or p, outer_q) torus knot, or -- when outer_q == 0 -- a
    plain circle of radius `circle_radius` wound p times so the rulings
    still line up."""
    inner = _torus_knot_at(p, q, t_inner, scale=knot_scale, tube=tube)
    inner[..., 2] *= inner_height
    inner[..., 2] += inner_lift
    if inner_rotation != 0.0:
        ca, sa = math.cos(inner_rotation), math.sin(inner_rotation)
        x, y = inner[..., 0].copy(), inner[..., 1].copy()
        inner[..., 0] = x * ca - y * sa
        inner[..., 1] = x * sa + y * ca
    if outer_q > 0:
        outer = _torus_knot_at(outer_p or p, outer_q, t_outer,
                               scale=outer_scale, tube=outer_tube)
        outer[..., 2] *= outer_height
    else:
        t = np.asarray(t_outer, dtype=float)
        outer = np.stack([circle_radius * np.cos(p * t),
                          circle_radius * np.sin(p * t),
                          np.zeros_like(t)], axis=-1)
    return inner, outer


def _knot_span_boundaries(p=2, q=3, knot_scale=1.0, tube=1.0,
                          inner_height=1.0, inner_lift=0.0,
                          inner_rotation=0.0, outer_p=0, outer_q=5,
                          outer_scale=2.0, outer_tube=1.0,
                          outer_height=1.0, circle_radius=4.5, m=96,
                          outer_phase=0.0):
    """Inner and outer boundary loops (each (m, 3)) for the knot span,
    sampled at m parameters around [0, 2pi) (`_knot_span_at`).
    `outer_phase` slides the outer samples along the outer loop, so
    sample i of the inner loop faces a point further on (the twist of
    the rulings); the loops themselves do not move."""
    t = np.linspace(0.0, _TWO_PI, m, endpoint=False)
    return _knot_span_at(t, t + outer_phase, p, q, knot_scale, tube,
                         inner_height, inner_lift, inner_rotation, outer_p,
                         outer_q, outer_scale, outer_tube, outer_height,
                         circle_radius)


def build_knot_span(p=2, q=3, knot_scale=1.0, tube=1.0,
                    inner_height=1.0, inner_lift=0.0, inner_rotation=0.0,
                    outer_p=0, outer_q=5, outer_scale=2.0,
                    outer_tube=1.0, outer_height=1.0, circle_radius=4.5,
                    res_u=96, res_v=16, shift=0.0):
    """Ruled surface between two concentric torus knots:
        S(u, v) = inner(u) (1 - v) + outer(u + shift) v,   v in [0, 1]
    the same straight-ruling interpolation as the stick hyperboloid,
    with the two coaxial circles replaced by an inner and an outer
    (p, q) torus knot.  `shift` is the right-handed family's twist."""
    inner, outer = _knot_span_boundaries(
        p, q, knot_scale, tube, inner_height, inner_lift, inner_rotation,
        outer_p, outer_q, outer_scale, outer_tube, outer_height,
        circle_radius, res_u, outer_phase=shift)
    v = np.linspace(0.0, 1.0, res_v + 1)
    P = np.empty((res_u, res_v + 1, 3))
    for j, vv in enumerate(v):
        P[:, j, :] = inner * (1.0 - vv) + outer * vv
    return _mesh_grid(P, wrap_u=True)


def rulings_knot_span(p=2, q=3, knot_scale=1.0, tube=1.0,
                      inner_height=1.0, inner_lift=0.0,
                      inner_rotation=0.0, outer_p=0, outer_q=5,
                      outer_scale=2.0, outer_tube=1.0, outer_height=1.0,
                      circle_radius=4.5, n=48, family='RIGHT',
                      shift=0.0):
    """The knot span's right family: n straight rods, rod i joining
    inner(u_i) to outer(u_i + shift).

    These are the surface's own rulings, which is why they are straight.
    The left family lies on the same surface and is curved in general
    (`left_rulings_knot_span`), so it is not returned here: `family`
    RIGHT or BOTH gives these rods, LEFT none."""
    if family not in ('RIGHT', 'BOTH'):
        return []
    inner, outer = _knot_span_boundaries(
        p, q, knot_scale, tube, inner_height, inner_lift, inner_rotation,
        outer_p, outer_q, outer_scale, outer_tube, outer_height,
        circle_radius, n, outer_phase=shift)
    return [(tuple(inner[i]), tuple(outer[i])) for i in range(n)]


def _knot_span_left(kw, n, shift):
    """What every left strand of the knot span shares: the strand starts
    u_j, each right ruling's twist Phi_j about the axis (the short way
    round), and the two radii of the turning profile -- the mean radius
    of the strands' starts on the inner knot and of their ends on the
    outer."""
    p = kw['p']
    po = kw['outer_p'] or p
    uj = _TWO_PI * np.arange(n) / n
    phi = po * (uj + shift) - p * uj - kw['inner_rotation']
    phi = (phi + math.pi) % _TWO_PI - math.pi
    u_end = uj - 2.0 * phi / p + shift
    a, _o = _knot_span_at(uj, uj, **kw)
    _i, b = _knot_span_at(u_end, u_end, **kw)
    return (uj, phi, float(np.hypot(a[:, 0], a[:, 1]).mean()),
            float(np.hypot(b[:, 0], b[:, 1]).mean()))


def _left_theta(v, phi, r1, r2):
    """How far a ruling of twist phi between radii r1 and r2 has turned
    about the axis a fraction v of the way along it."""
    return np.arctan2(v * r2 * np.sin(phi),
                      (1.0 - v) * r1 + v * r2 * np.cos(phi))


def left_rulings_knot_span(p=2, q=3, knot_scale=1.0, tube=1.0,
                           inner_height=1.0, inner_lift=0.0,
                           inner_rotation=0.0, outer_p=0, outer_q=5,
                           outer_scale=2.0, outer_tube=1.0,
                           outer_height=1.0, circle_radius=4.5, n=48,
                           shift=0.0, samples=256):
    """The knot span's left family: n strands laid on the surface
        S(u, v) = (1 - v) inner(u) + v outer(u + shift),
    each returned as a (samples + 1, 3) polyline from the inner knot to
    the outer.

    A stick hyperboloid's two families mirror each other.  The right
    ruling leaving a point of the bottom circle turns about the axis, by
    the time it is a fraction v of the way up, through
        theta(v) = arg((1 - v) r1 + v r2 e^{i Phi}),
    r1 and r2 the radii of its two ends and Phi the twist between them;
    the left ruling leaving the same point turns back through the same
    theta(v).  Both lie on one hyperboloid, so the left ruling's point at
    v is the surface point S(u, v) whose right ruling has turned FORWARD
    through theta to reach it: with the inner curve's azimuth p u + rho,
        p u + rho + theta = p u_j + rho - theta,  so u = u_j - 2 theta / p.

    That is the construction here: left strand j is
        v -> S(u_j - 2 theta(v) / p, v),
    with one turning profile theta for every strand, taken from the mean
    radii of the strands' two ends (`_knot_span_left`).  On two coaxial
    circles the radii are constant, so it reproduces the hyperboloid's
    straight left rulings exactly.  On knotted rails no surface but a quadric carries
    two straight families, so the strands bend -- but they lie on the
    same surface as the right rulings, fill it in with them as n grows,
    and cross them on it, exactly, wherever u_j - 2 theta_j / p passes
    some u_i.  Each strand ends at outer(u_j - shift + 2 rho / p), rho
    being the inner rotation: the right ruling's twist, mirrored.

    Two nearby rules fail, and say why this one is shaped as it is.
    Taking the radii at each point along the strand makes the rule
    implicit -- u appears on both sides -- and on folded knots that
    equation has no continuous solution: the strand jumps across the
    surface.  Taking each strand's OWN end radii keeps it explicit, but
    where the knot's radius changes quickly neighbouring strands turn at
    different rates, overtake one another, and meet a right ruling out of
    order; a weave can then no longer alternate there.  With one shared
    profile the strands are translates of each other in u, so they never
    cross one another and the two families form the same regular lattice
    as a hyperboloid's.
    """
    kw = dict(p=p, q=q, knot_scale=knot_scale, tube=tube,
              inner_height=inner_height, inner_lift=inner_lift,
              inner_rotation=inner_rotation, outer_p=outer_p,
              outer_q=outer_q, outer_scale=outer_scale,
              outer_tube=outer_tube, outer_height=outer_height,
              circle_radius=circle_radius)
    uj, phi, r1, r2 = _knot_span_left(kw, n, shift)
    v = np.linspace(0.0, 1.0, samples + 1)[:, None]
    U = uj[None, :] - 2.0 * _left_theta(v, phi, r1, r2) / p
    inner, outer = _knot_span_at(U, U + shift, **kw)
    pts = (1.0 - v)[..., None] * inner + v[..., None] * outer
    return [pts[:, j].copy() for j in range(n)]


def knot_span_weave(p=2, q=3, knot_scale=1.0, tube=1.0, inner_height=1.0,
                    inner_lift=0.0, inner_rotation=0.0, outer_p=0,
                    outer_q=5, outer_scale=2.0, outer_tube=1.0,
                    outer_height=1.0, circle_radius=4.5, n=48, shift=0.0,
                    overhang=0.0, samples=256):
    """Both families of the knot span ready to weave: (right strands,
    left strands, crossings, neighbour gap) for `weave_rulings`.

    The right strands are the straight rulings, the left strands the
    curves of `left_rulings_knot_span`.  An overhang carries both on past
    the knots along the surface itself -- a right ruling straight on, a
    left strand by running its v beyond [0, 1] -- by one distance,
    `overhang` times the longest strand, so the weave carries on as well.

    The crossings are exact rather than searched for.  Right ruling i
    lies at u = u_i and left strand j at u = u_j - 2 theta_j(v) / p, so
    they meet where theta_j(v) = p (u_j - u_i) / 2, with u_i taken round
    the knot as many times as the strand's sweep allows; and
        tan theta = v r2 sin Phi / ((1 - v) r1 + v r2 cos Phi)
    gives that v in closed form,
        v = r1 sin theta / (r2 sin(Phi - theta) + r1 sin theta),
    r1 and r2 being the shared profile radii (`_knot_span_left`).
    The crossing point is S(u_i, v), on both strands, and the normal
    there is the cross product of the two strands' tangents.

    The gap handed on is the 10th percentile of the spacing between
    neighbouring right rulings, not the smallest: where the surface
    folds, neighbours of one family come within a hair of each other, and
    sizing every ribbon to that one place would leave them all too thin
    to see.
    """
    kw = dict(p=p, q=q, knot_scale=knot_scale, tube=tube,
              inner_height=inner_height, inner_lift=inner_lift,
              inner_rotation=inner_rotation, outer_p=outer_p,
              outer_q=outer_q, outer_scale=outer_scale,
              outer_tube=outer_tube, outer_height=outer_height,
              circle_radius=circle_radius)
    uj, phi, r1, r2 = _knot_span_left(kw, n, shift)

    def left_u(v, j=slice(None)):
        return uj[j] - 2.0 * _left_theta(v, phi[j], r1, r2) / p

    def left_point(v, j=slice(None)):
        U = left_u(v, j)
        inner, outer = _knot_span_at(U, U + shift, **kw)
        return (1.0 - v)[..., None] * inner + v[..., None] * outer

    # lengths before any overhang, to turn one distance into v on each
    A0, A1 = _knot_span_at(uj, uj + shift, **kw)
    LA = np.linalg.norm(A1 - A0, axis=1)
    grid = np.linspace(0.0, 1.0, samples + 1)[:, None]
    LB = np.linalg.norm(np.diff(left_point(grid * np.ones(n)), axis=0),
                        axis=-1).sum(axis=0)
    amount = overhang * float(max(LA.max(), LB.max()))
    hA, hB = amount / LA, amount / LB

    right = [np.stack([A0[i] - hA[i] * (A1[i] - A0[i]),
                       A1[i] + hA[i] * (A1[i] - A0[i])]) for i in range(n)]
    V = -hB[None, :] + (1.0 + 2.0 * hB)[None, :] * grid          # (K+1, n)
    LP = left_point(V)
    cum = np.concatenate([np.zeros((1, n)), np.cumsum(
        np.linalg.norm(np.diff(LP, axis=0), axis=-1), axis=0)], axis=0)
    frac = cum / cum[-1][None, :]
    left = [LP[:, j].copy() for j in range(n)]

    ia, ib, vv = [], [], []
    eps = 1e-12
    for j in range(n):
        v_lo, v_hi = -hB[j], 1.0 + hB[j]
        ua, ub = left_u(np.array(v_lo), j), left_u(np.array(v_hi), j)
        umin, umax = min(ua, ub), max(ua, ub)
        m_lo = np.ceil((umin - eps - uj) / _TWO_PI).astype(int)
        m_hi = np.floor((umax + eps - uj) / _TWO_PI).astype(int)
        for m in range(int(m_lo.min()), int(m_hi.max()) + 1):
            ii = np.nonzero((m_lo <= m) & (m <= m_hi))[0]
            theta = p * (uj[j] - (uj[ii] + _TWO_PI * m)) / 2.0
            den = r2 * np.sin(phi[j] - theta) + r1 * np.sin(theta)
            ok = np.abs(den) > 1e-12
            v = np.where(ok, r1 * np.sin(theta) / np.where(ok, den, 1.0),
                         np.nan)
            ok &= ((v >= v_lo - 1e-9) & (v <= v_hi + 1e-9)
                   & (v >= -hA[ii] - 1e-9) & (v <= 1.0 + hA[ii] + 1e-9))
            ia.extend(ii[ok])
            ib.extend([j] * int(ok.sum()))
            vv.extend(v[ok])
    ia, ib, vv = np.asarray(ia, dtype=int), np.asarray(ib, dtype=int), \
        np.asarray(vv, dtype=float)
    point = (1.0 - vv)[:, None] * A0[ia] + vv[:, None] * A1[ia]
    ta = (vv + hA[ia]) / (1.0 + 2.0 * hA[ia])
    tb = np.array([np.interp(v, V[:, j], frac[:, j])
                   for v, j in zip(vv, ib)])
    TA = (A1[ia] - A0[ia]) / LA[ia, None]
    dv = 1e-6
    TB = np.stack([left_point(np.array([v + dv]), j)[0]
                   - left_point(np.array([v - dv]), j)[0]
                   for v, j in zip(vv, ib)]) if len(vv) else np.zeros((0, 3))
    TB /= np.maximum(np.linalg.norm(TB, axis=1, keepdims=True), 1e-300)
    normal = np.cross(TA, TB)
    sin = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(sin, 1e-300)[:, None]
    crossings = dict(ia=ia, ta=ta, ib=ib, tb=tb, point=point,
                     normal=normal, sin=sin)
    spacing = [segment_distance(right[i][0], right[i][1],
                                right[(i + 1) % n][0], right[(i + 1) % n][1])
               for i in range(n)]
    return right, left, crossings, float(np.percentile(spacing, 10))


# --------------------------------------------------------------------
# 2. compound helical cone (Solomonic column)
# --------------------------------------------------------------------

def build_helical_cone(base_radius=1.0, height=2.0, flutes=6,
                       flute_depth=0.18, twist=2.0, taper=1.0,
                       orbit_amp=0.0, orbit_turns=1.0,
                       res_u=120, res_v=80):
    """Compound helical cone.  A conical envelope R(t)=R0(1-t)^taper is
    fluted by `flutes` ridges that wind helically at rate `twist`
    (turns over the full height); the flutes fade toward the apex
    (ornament -> arris).  An optional planetary helix of amplitude
    `orbit_amp` and `orbit_turns` sweeps the section centre around a
    second helix.  t runs base(0) -> apex(1); the apex is one point."""
    theta = np.linspace(0.0, _TWO_PI, res_u, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)
    rings = []
    for j in range(res_v):
        t = j / res_v
        Rt = base_radius * (1.0 - t) ** max(taper, 1e-6)
        eps = flute_depth * (1.0 - t)          # flutes fade to arris
        r = Rt * (1.0 + eps * np.cos(flutes * theta
                                     + _TWO_PI * twist * t))
        cx = orbit_amp * Rt * math.cos(_TWO_PI * orbit_turns * t)
        cy = orbit_amp * Rt * math.sin(_TWO_PI * orbit_turns * t)
        ring = np.stack([cx + r * ct, cy + r * st,
                         np.full(res_u, height * t)], axis=1)
        rings.append(ring)
    return _capped(rings, (0.0, 0.0, height))


# --------------------------------------------------------------------
# 3. spiral ruled surface (Farris)
# --------------------------------------------------------------------

def arrises_helical_cone(base_radius=1.0, height=2.0, flutes=6,
                         flute_depth=0.18, twist=2.0, taper=1.0,
                         orbit_amp=0.0, orbit_turns=1.0, n=48):
    """Segments along the compound helical cone's ARRISES.

    Every other mode here is straight-ruled, and its rod output draws
    the rulings.  This one is not: the flutes wind as they climb and the
    radius falls off faster than linearly, so no straight line lies in
    the surface.  What it does have is its `flutes` ridge curves, the
    arrises running from base to apex, and those are the lines a
    Solomonic column is actually read by -- so they are what the rod and
    curve outputs draw here.  Each is emitted as a polyline of short
    segments rather than one straight rod.

    A ridge sits where cos(flutes.theta + 2pi.twist.t) = 1, so
    theta_k(t) = 2pi(k - twist.t) / flutes.
    """
    m = max(1, int(flutes))
    steps = max(8, int(n))
    segs = []
    for k in range(m):
        pts = []
        for j in range(steps + 1):
            t = j / steps
            th = _TWO_PI * (k - twist * t) / m
            Rt = base_radius * (1.0 - t) ** max(taper, 1e-6)
            r = Rt * (1.0 + flute_depth * (1.0 - t))
            cx = orbit_amp * Rt * math.cos(_TWO_PI * orbit_turns * t)
            cy = orbit_amp * Rt * math.sin(_TWO_PI * orbit_turns * t)
            pts.append((cx + r * math.cos(th), cy + r * math.sin(th),
                        height * t))
        segs.extend(zip(pts, pts[1:]))
    return segs


def build_spiral_ruled(tightness=0.15, slope=1.0, turns=2.0,
                       petals=1, petal_amp=0.0, v_extent=1.0,
                       res_u=200, res_v=20):
    """Farris' spiral ruled surface: base curve
        b(u) = e^{k u} (cos u + A cos(p u),  sin u + A sin(p u))
    (a logarithmic spiral for p=1, A=0; an n-fold rosette otherwise),
    swept as S = b(u) + v b'(u) horizontally with z = slope * v.  For
    tightness k=0, petals=1, A=0 this is a hyperboloid."""
    k = tightness
    u = np.linspace(0.0, _TWO_PI * turns, res_u)
    e = np.exp(k * u)
    p, A = float(petals), petal_amp
    # base curve f(u) and its derivative, then b = e*f, b' = e(k f + f')
    fx = np.cos(u) + A * np.cos(p * u)
    fy = np.sin(u) + A * np.sin(p * u)
    fpx = -np.sin(u) - A * p * np.sin(p * u)
    fpy = np.cos(u) + A * p * np.cos(p * u)
    bx, by = e * fx, e * fy
    dbx = e * (k * fx + fpx)
    dby = e * (k * fy + fpy)
    v = np.linspace(-v_extent, v_extent, res_v + 1)
    P = np.empty((res_u, res_v + 1, 3))
    for j, vv in enumerate(v):
        P[:, j, 0] = bx + vv * dbx
        P[:, j, 1] = by + vv * dby
        P[:, j, 2] = slope * vv
    return _mesh_grid(P)


def _spiral_frame(u, tightness, petals, petal_amp):
    """Base point b(u) and horizontal ruling direction b'(u) of the
    spiral ruled surface, as (bx, by, dbx, dby) arrays shaped like u."""
    k = tightness
    u = np.asarray(u, dtype=float)
    e = np.exp(k * u)
    p, A = float(petals), petal_amp
    fx = np.cos(u) + A * np.cos(p * u)
    fy = np.sin(u) + A * np.sin(p * u)
    fpx = -np.sin(u) - A * p * np.sin(p * u)
    fpy = np.cos(u) + A * p * np.cos(p * u)
    return e * fx, e * fy, e * (k * fx + fpx), e * (k * fy + fpy)


def _spiral_point(u, v, tightness, slope, petals, petal_amp):
    """S(u, v) = (b(u) + v b'(u), slope v), broadcast over u and v."""
    bx, by, dbx, dby = _spiral_frame(u, tightness, petals, petal_amp)
    v = np.asarray(v, dtype=float)
    return np.stack(np.broadcast_arrays(bx + v * dbx, by + v * dby,
                                        slope * v), axis=-1)


def rulings_spiral(tightness=0.15, slope=1.0, turns=2.0, petals=1,
                   petal_amp=0.0, v_extent=1.0, n=64, family='RIGHT'):
    """The spiral surface's own straight rulings, spread evenly over the
    turns and running v = -v_extent .. v_extent.  They are its right
    family, so RIGHT or BOTH gives them and LEFT none: the left family
    lies on the surface as curves (`left_rulings_spiral`)."""
    if family not in ('RIGHT', 'BOTH'):
        return []
    u = np.linspace(0.0, _TWO_PI * turns, n)
    lo = _spiral_point(u, -v_extent, tightness, slope, petals, petal_amp)
    hi = _spiral_point(u, v_extent, tightness, slope, petals, petal_amp)
    return [(tuple(lo[i]), tuple(hi[i])) for i in range(n)]


def _spiral_theta(v, tightness):
    """How far a right ruling of the log spiral has turned about the axis
    a distance v along it, from its base point:
        theta(v) = arg(1 + v (k + i)),
    because b + v b' = b (1 + v (k + i)) for b = e^{(k + i) u}.  It rises
    with v for every k (d theta / dv = 1 / |1 + v (k + i)|^2)."""
    return np.arctan2(v, 1.0 + tightness * np.asarray(v, dtype=float))


def _spiral_theta_inverse(delta, tightness, v_lo, v_hi):
    """The v in [v_lo, v_hi] where theta(v) = delta, held at an end where
    theta does not reach delta.  theta(v) = delta means (v, 1 + k v) =
    lam (sin delta, cos delta) for some lam > 0, so
        v = sin delta / (cos delta - k sin delta)."""
    delta = np.asarray(delta, dtype=float)
    t_lo = _spiral_theta(v_lo, tightness)
    t_hi = _spiral_theta(v_hi, tightness)
    den = np.cos(delta) - tightness * np.sin(delta)
    inside = (delta > t_lo) & (delta < t_hi) & (den > 0.0)
    v = np.where(inside, np.sin(delta) / np.where(inside, den, 1.0), 0.0)
    return np.where(delta <= t_lo, v_lo, np.where(delta >= t_hi, v_hi, v))


def _spiral_left(tightness, turns, n, v_lo, v_hi):
    """The spiral's left strands: their starts u_j, spaced like the n
    right rulings and reaching past both ends of the turns far enough to
    fill the surface, and the v-interval [a_j, b_j] of each that stays on
    it (0 <= u_j - 2 theta(v) <= 2 pi turns)."""
    U = _TWO_PI * turns
    du = U / max(1, n - 1)
    th_lo = float(_spiral_theta(v_lo, tightness))
    th_hi = float(_spiral_theta(v_hi, tightness))
    # u(v) = u_j - 2 theta(v) falls from u_j - 2 th_lo to u_j - 2 th_hi
    j0 = int(math.ceil(2.0 * th_lo / du - 1e-9))
    j1 = int(math.floor((U + 2.0 * th_hi) / du + 1e-9))
    uj = du * np.arange(j0, j1 + 1)
    a = _spiral_theta_inverse((uj - U) / 2.0, tightness, v_lo, v_hi)
    b = _spiral_theta_inverse(uj / 2.0, tightness, v_lo, v_hi)
    keep = b - a > 1e-9 * max(1.0, v_hi - v_lo)
    return uj[keep], a[keep], b[keep]


def left_rulings_spiral(tightness=0.15, slope=1.0, turns=2.0, petals=1,
                        petal_amp=0.0, v_extent=1.0, n=64, samples=128):
    """The spiral ruled surface's left family, as polylines lying on it.

    A log spiral b(u) = e^{(k + i) u} is carried onto itself by turning
    and scaling, and b + v b' = b (1 + v (k + i)): the right ruling from
    b(u_j) has turned about the axis, a distance v along, through
    theta(v) = arg(1 + v (k + i)) (`_spiral_theta`) -- the same for every
    ruling.  The left strand mirrors it, turning BACK through theta(v),
    and the surface point it reaches is the one whose own right ruling
    turned forward to get there:
        u + theta(v) = u_j - theta(v),   so u(v) = u_j - 2 theta(v),
    and left strand j is v -> S(u_j - 2 theta(v), v).  With no tightness
    and no rosette the surface is a hyperboloid and each strand is its
    straight left ruling b(u_j) (1 - i v); otherwise the strands bend, but
    they lie on the same surface, each crosses every right ruling it
    reaches exactly once, and all are one curve shifted in u, so the two
    families form a regular lattice.  A rosette base curve is not
    self-similar, so there the strands still lie on the surface and keep
    the lattice but no longer mirror their partners exactly.

    The strands are spaced like the right rulings, reach past both ends
    of the turns far enough to fill the surface, and are clipped to it.
    """
    uj, a, b = _spiral_left(tightness, turns, n, -v_extent, v_extent)
    t = np.linspace(0.0, 1.0, samples + 1)[:, None]
    v = a[None, :] + (b - a)[None, :] * t
    pts = _spiral_point(uj[None, :] - 2.0 * _spiral_theta(v, tightness), v,
                        tightness, slope, petals, petal_amp)
    return [pts[:, j].copy() for j in range(len(uj))]


def spiral_weave(tightness=0.15, slope=1.0, turns=2.0, petals=1,
                 petal_amp=0.0, v_extent=1.0, n=64, overhang=0.0,
                 samples=128):
    """Both families of the spiral ruled surface ready to weave: (right
    strands, left strands, crossings, neighbour gap) for `weave_rulings`,
    as `knot_span_weave` gives them for the knots.

    An overhang runs every strand on past both rails in v, by `overhang`
    times the length of its ruling, so every end stays level with the
    new rails v = +-v_extent (1 + 2 overhang) -- on a surface that grows
    as it winds, one distance for all would leave the inner turns'
    ribbons sticking far out and the outer ones barely clear.

    The crossings are exact.  Right ruling i lies at u = u_i and left
    strand j at u = u_j - 2 theta(v), so they meet where
        theta(v) = (u_j - u_i) / 2,
    which `_spiral_theta_inverse` solves in closed form; the crossing is
    the surface point S(u_i, v), on both strands.  The gap handed on is
    the 10th percentile of the spacing between neighbouring right
    rulings.
    """
    U = _TWO_PI * turns
    V = v_extent * (1.0 + 2.0 * overhang)
    ui = np.linspace(0.0, U, n)
    lo = _spiral_point(ui, -V, tightness, slope, petals, petal_amp)
    hi = _spiral_point(ui, V, tightness, slope, petals, petal_amp)
    right = [np.stack([lo[i], hi[i]]) for i in range(n)]
    uj, a, b = _spiral_left(tightness, turns, n, -V, V)
    m = len(uj)

    def left_point(v, j):
        return _spiral_point(uj[j] - 2.0 * _spiral_theta(v, tightness), v,
                             tightness, slope, petals, petal_amp)

    t = np.linspace(0.0, 1.0, samples + 1)[:, None]
    Vs = a[None, :] + (b - a)[None, :] * t                     # (K+1, m)
    LP = left_point(Vs, np.arange(m)[None, :])
    cum = np.concatenate([np.zeros((1, m)), np.cumsum(
        np.linalg.norm(np.diff(LP, axis=0), axis=-1), axis=0)], axis=0)
    frac = cum / np.maximum(cum[-1][None, :], 1e-300)
    left = [LP[:, j].copy() for j in range(m)]

    # right ruling i meets strand j where theta = (u_j - u_i) / 2, if
    # that happens on the part of the strand kept on the surface
    delta = (uj[None, :] - ui[:, None]) / 2.0                  # (n, m)
    ok = ((delta >= _spiral_theta(a, tightness)[None, :] - 1e-12)
          & (delta <= _spiral_theta(b, tightness)[None, :] + 1e-12))
    ia, ib = np.nonzero(ok)
    vv = _spiral_theta_inverse(delta[ia, ib], tightness, a[ib], b[ib])
    point = _spiral_point(ui[ia], vv, tightness, slope, petals, petal_amp)
    ta = (vv + V) / (2.0 * V)
    tb = np.array([np.interp(v, Vs[:, j], frac[:, j])
                   for v, j in zip(vv, ib)])
    TA = hi[ia] - lo[ia]
    TA /= np.maximum(np.linalg.norm(TA, axis=1, keepdims=True), 1e-300)
    dv = 1e-6
    TB = left_point(vv + dv, ib) - left_point(vv - dv, ib)
    TB /= np.maximum(np.linalg.norm(TB, axis=1, keepdims=True), 1e-300)
    normal = np.cross(TA, TB)
    sin = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(sin, 1e-300)[:, None]
    crossings = dict(ia=ia, ta=ta, ib=ib, tb=tb, point=point,
                     normal=normal, sin=sin)
    spacing = [segment_distance(right[i][0], right[i][1],
                                right[i + 1][0], right[i + 1][1])
               for i in range(n - 1)] or [1.0]
    return right, left, crossings, float(np.percentile(spacing, 10))


# --------------------------------------------------------------------
# basket weave: rulings as stakes, the curves along the surface as
# weavers -- woven ribbons on a surface with one ruling family
# --------------------------------------------------------------------

def _weft_chart(S, n, span, closure, v, fixed=(False, False)):
    """One piece of surface for `weft_weave`.

    S(u, v) is the surface, straight in v, so its rulings are the lines
    u = const; `span` is its u range and `v` = (v0, v1) the stretch of
    each ruling woven.  `closure` says how the piece closes round in u:
    None (open), 'loop' (S(u + P, v) = S(u, v), P the span) or 'mobius'
    (S(u + P, v) = S(u, -v), with v0 = -v1: a half-twisted band).
    `fixed` marks the v ends where rulings run together, which an
    overhang must not push further into.

    The n rulings are spread evenly (n made even on a loop, where a plain
    weave needs an even count to close round), and the weavers sit at
    heights spaced like the rulings, so the cells come out roughly
    square.  On a Mobius band each weaver meets its mirror image after
    one lap, so the plain weave closes only when the weaver count and
    the ruling count differ in parity; the weaver count takes it."""
    u0, u1 = span
    n = max(2, int(n))
    if closure == 'loop' and n % 2:
        n += 1
    if closure:
        u = u0 + (u1 - u0) * np.arange(n) / n
    else:
        u = np.linspace(u0, u1, n)
    v0, v1 = v
    ends = [S(u, np.full(n, vv)) for vv in (v0, 0.5 * (v0 + v1), v1)]
    step = np.linalg.norm(np.diff(ends[1], axis=0), axis=1)
    length = np.linalg.norm(ends[2] - ends[0], axis=1)
    pitch = float(np.median(step)) if len(step) else 1.0
    m = int(np.clip(round(float(np.median(length)) / max(pitch, 1e-12)),
                    2, 400))
    if closure == 'mobius' and (m - n) % 2 == 0:
        m += 1
    weft = v0 + (np.arange(m) + 0.5) * (v1 - v0) / m
    return dict(S=S, u=u, span=(u0, u1), closure=closure, v=(v0, v1),
                weft=weft, fixed=tuple(fixed))


def _surface_du(S, u, v, h=1e-6):
    """dS/du by central difference."""
    return (S(u + h, v) - S(u - h, v)) / (2.0 * h)


def weft_weave(charts, overhang=0.0, sub=6):
    """Woven ribbons for a surface with one family of rulings: (rulings,
    weavers, crossings, neighbour gap) for `weave_rulings`, from the
    pieces of surface `_weft_chart` describes.

    A ruling is the line u = u_i; a weaver is the curve v = v_k running
    along the surface, a copy of the base curve.  Every weaver crosses
    every ruling of its piece exactly once per lap, at the surface point
    S(u_i, v_k), so the crossings are exact and form a lattice.  On a
    loop the weaver closes on itself; on a Mobius band it runs two laps,
    the second at -v_k, before it does.

    An overhang runs the rulings on past their free ends -- straight,
    along the surface -- by `overhang` times the longest ruling, and the
    open weavers past the first and last rulings by the same distance.
    Open weavers always reach half a ruling spacing past the end rulings,
    so they do not stop dead on a crossing.  The gap handed on is the
    10th percentile of the spacing between neighbouring rulings.
    """
    right, left = [], []
    parts = {k: [] for k in ('ia', 'ta', 'ib', 'tb', 'point', 'TA', 'TB')}
    spacing = []
    longest = 0.0
    for ch in charts:
        n = len(ch['u'])
        a = ch['S'](ch['u'], np.full(n, ch['v'][0]))
        b = ch['S'](ch['u'], np.full(n, ch['v'][1]))
        longest = max(longest, float(np.linalg.norm(b - a, axis=1).max()))
    amount = overhang * longest
    for ch in charts:
        S, u, closure = ch['S'], ch['u'], ch['closure']
        n = len(u)
        v0, v1 = ch['v']
        fix0, fix1 = ch['fixed']
        a = S(u, np.full(n, v0))
        b = S(u, np.full(n, v1))
        L = np.maximum(np.linalg.norm(b - a, axis=1), 1e-300)
        per = (v1 - v0) / L                          # v per unit length
        lo = v0 - (0.0 if fix0 else amount) * per
        hi = v1 + (0.0 if fix1 else amount) * per
        r0 = len(right)
        A_, B_ = S(u, lo), S(u, hi)
        right.extend(np.stack([A_[i], B_[i]]) for i in range(n))
        TA = (b - a) / L[:, None]
        for i in range(n if closure else n - 1):
            k = (i + 1) % n
            spacing.append(segment_distance(a[i], b[i], a[k], b[k]))

        u0, u1 = ch['span']
        period = u1 - u0
        du_step = period / n if closure else (u[-1] - u[0]) / max(1, n - 1)
        for vk in ch['weft']:
            if closure == 'mobius' and vk < -1e-12:
                continue              # the weaver at -vk is this one's lap 2
            laps = 2 if closure == 'mobius' and vk > 1e-12 else 1
            if closure:
                cu = np.concatenate([u + lap * period for lap in range(laps)])
                idx = np.tile(np.arange(n), laps)
                vr = np.concatenate([
                    np.full(n, vk * (-1.0 if closure == 'mobius' and lap
                                     else 1.0)) for lap in range(laps)])
                start, end = u0, u0 + laps * period
            else:
                cu, idx, vr = u.copy(), np.arange(n), np.full(n, vk)
                reach = [amount / max(float(np.linalg.norm(
                    _surface_du(S, np.array(uu), np.array(vk)))), 1e-300)
                    for uu in (u[0], u[-1])]
                start = u[0] - 0.5 * du_step - reach[0]
                end = u[-1] + 0.5 * du_step + reach[1]
            grid = np.union1d(np.linspace(start, end,
                                          laps * max(n, 2) * sub + 1), cu)
            pts = S(grid, np.full(len(grid), vk))
            if closure:
                pts[-1] = pts[0]
            cum = np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            if cum[-1] < 1e-12:
                continue
            j = len(left)
            left.append(pts)
            parts['ia'].append(r0 + idx)
            parts['ta'].append((vr - lo[idx]) / (hi[idx] - lo[idx]))
            parts['ib'].append(np.full(len(cu), j))
            parts['tb'].append(np.interp(cu, grid, cum) / cum[-1])
            parts['point'].append(S(u[idx], vr))
            parts['TA'].append(TA[idx])
            parts['TB'].append(_surface_du(S, cu, np.full(len(cu), vk)))
    if parts['ia']:
        cat = {k: np.concatenate(v) for k, v in parts.items()}
    else:
        cat = dict(ia=np.zeros(0, int), ta=np.zeros(0), ib=np.zeros(0, int),
                   tb=np.zeros(0), point=np.zeros((0, 3)),
                   TA=np.zeros((0, 3)), TB=np.zeros((0, 3)))
    TB = cat['TB'] / np.maximum(np.linalg.norm(cat['TB'], axis=1,
                                               keepdims=True), 1e-300)
    normal = np.cross(cat['TA'], TB)
    sin = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(sin, 1e-300)[:, None]
    crossings = dict(ia=cat['ia'].astype(int), ta=cat['ta'],
                     ib=cat['ib'].astype(int), tb=cat['tb'],
                     point=cat['point'], normal=normal, sin=sin)
    gap = float(np.percentile(spacing, 10)) if spacing else 1.0
    return right, left, crossings, gap


# --------------------------------------------------------------------
# 4. conoids  (right conoid S = (v cos u, v sin u, h(u)))
# --------------------------------------------------------------------

def named_ruled_curves(kind, u, amp=0.5, extent=1.0, folds=2,
                       wallis_a=1.0, wallis_b=0.6):
    """The two directrix curves A(u), B(u) of a named ruled surface.

    Every one of these is "the union of the lines A(u) B(u)" for two
    curves, which is the oldest way of writing a ruled surface down and
    the way all five sources state them.  Returning the pair rather than
    a mesh keeps the definition in one place: the surface, its rulings
    and its self-test all read the same two curves.
    """
    u = np.asarray(u, dtype=float)
    z = np.zeros_like(u)
    if kind == 'GAUDI':
        # Right conoid z = k x sin(y/a): rulings run along x at each
        # height y, all meeting the y-axis at right angles.  Gaudi
        # roofed the Sagrada Familia escoles with it -- a warped surface
        # you can nevertheless build out of straight timber.
        a = max(1e-6, wallis_a)
        k = amp / max(1e-6, extent)
        return (np.stack([-extent + z, u, -k * extent * np.sin(u / a)], -1),
                np.stack([extent + z, u, k * extent * np.sin(u / a)], -1))
    if kind == 'GUIMARD':
        # M moves along a line with a sinusoidal law, N around a circle
        # with a doubled sinusoid, and the segment MN sweeps the surface
        # Guimard used in his Art Nouveau ironwork.
        a, b, c = wallis_a, extent, amp
        return (np.stack([a * np.cos(u), z, z], -1),
                np.stack([b * np.cos(u), b * np.sin(u),
                          c * np.sin(u) ** 2], -1))
    if kind == 'MILK_CARTON':
        # The "berlingot": every horizontal section is an ellipse whose
        # axes trade places as the height rises, so the top and bottom
        # degenerate into two perpendicular segments.  Linear in the
        # ruling parameter, which is why a paper carton folds flat.
        k, al = max(1e-6, amp), max(1e-6, extent)
        return (np.stack([z, 2 * k * al * np.sin(u), -al + z], -1),
                np.stack([2 * k * al * np.cos(u), z, al + z], -1))
    if kind == 'RULED_CUBIC':
        # A conic, a line meeting its plane, and a homography between
        # them: join corresponding points and the union is a cubic
        # surface.  Here the conic is the unit circle taken rationally
        # and the line is Oz, with the identity homography.
        s = np.tan(np.clip(u, -1.5, 1.5))
        d = 1.0 + s * s
        return (np.stack([(1.0 - s * s) / d, 2.0 * s / d, z], -1),
                np.stack([z, z, extent * s], -1))
    # CONSTANT_SLOPE -- Monge's sandpile.  Rise from each point of a
    # closed base curve along the curve's outward normal at a fixed
    # angle; the result is the shape a heap of dry sand takes, and it is
    # developable, since a constant-slope surface is the envelope of a
    # one-parameter family of equal cones.
    n = max(2, int(folds))
    r = 1.0 + wallis_b * np.cos(n * u)               # base curve radius
    dr = -wallis_b * n * np.sin(n * u)
    cu, su = np.cos(u), np.sin(u)
    base = np.stack([r * cu, r * su, z], -1)
    # outward normal of the polar curve r(u)
    tx, ty = dr * cu - r * su, dr * su + r * cu
    ln = np.sqrt(tx * tx + ty * ty)
    nx, ny = ty / ln, -tx / ln
    slope = math.tan(math.radians(min(85.0, max(5.0, wallis_a * 45.0))))
    top = base + extent * np.stack([nx, ny, slope * np.ones_like(u)], -1)
    return base, top


def build_named_ruled(kind='GAUDI', amp=0.5, extent=1.0, folds=2,
                      wallis_a=1.0, wallis_b=0.6, res_u=160, res_v=16):
    """Mesh a named ruled surface as the lines joining its two curves."""
    wrap = kind in ('GUIMARD', 'MILK_CARTON', 'CONSTANT_SLOPE')
    if kind == 'GAUDI':
        u = np.linspace(-math.pi * wallis_a, math.pi * wallis_a, res_u)
    elif kind == 'RULED_CUBIC':
        u = np.linspace(-1.45, 1.45, res_u)
    else:
        u = np.linspace(0.0, _TWO_PI, res_u, endpoint=not wrap)
    A, B = named_ruled_curves(kind, u, amp, extent, folds,
                              wallis_a, wallis_b)
    v = np.linspace(0.0, 1.0, res_v + 1)
    P = A[:, None, :] + v[None, :, None] * (B - A)[:, None, :]
    return _mesh_grid(P, wrap_u=wrap)


def rulings_named_ruled(kind='GAUDI', amp=0.5, extent=1.0, folds=2,
                        wallis_a=1.0, wallis_b=0.6, n=48):
    if kind == 'GAUDI':
        u = np.linspace(-math.pi * wallis_a, math.pi * wallis_a, n)
    elif kind == 'RULED_CUBIC':
        u = np.linspace(-1.45, 1.45, n)
    else:
        u = np.linspace(0.0, _TWO_PI, n, endpoint=False)
    A, B = named_ruled_curves(kind, u, amp, extent, folds,
                              wallis_a, wallis_b)
    return [(tuple(A[i]), tuple(B[i])) for i in range(len(u))]


def _cone_over(res_v, extent, cx, cy, cz, wrap=True):
    """Mesh the cone with apex at the origin over the directrix
    (cx, cy, cz)(v): a welded apex vertex, then rings scaled u/m out to
    the directrix at u = extent.  Faces wind consistently, so an
    orientation sweep sees a plain disk (wrap=False) or cone."""
    n = len(cx)
    verts = [(0.0, 0.0, 0.0)]
    rows = []
    m = max(2, int(res_v))
    for j in range(1, m + 1):
        u = extent * j / m
        base = len(verts)
        verts.extend(zip(u * cx, u * cy, u * cz))
        rows.append(list(range(base, base + n)))
    kmax = n if wrap else n - 1
    faces = []
    for i in range(kmax):
        i1 = (i + 1) % n
        faces.append((0, rows[0][i], rows[0][i1]))
    for j in range(m - 1):
        for i in range(kmax):
            i1 = (i + 1) % n
            faces.append((rows[j][i], rows[j + 1][i],
                          rows[j + 1][i1], rows[j][i1]))
    return verts, faces


def build_conoid(kind='PLUCKER', amp=0.5, folds=2, wallis_a=1.0,
                 wallis_b=0.6, extent=1.0, res_u=120, res_v=16,
                 turns=2.0):
    """Right conoids, ruled cones and the Whitney umbrella.
        PLUCKER  h = amp sin(2u)            (Plucker's cylindroid)
        NFOLD    h = amp sin(folds*u)       (n-leaved conoid)
        WALLIS   h = amp sqrt(a^2 - b^2 cos^2 u)   (Wallis conical edge)
        ZINDLER  h = amp tan(folds*u)       (Zindler's conoid)
        WHITNEY  S = (u v, u, v^2)          (pinch-point umbrella)
        PARABOLIC_CONOID  S = (v, u, amp v (1 - u^2)), the roof-shell
                 conoid a^2 z = x(b^2 - y^2) with b = 1, amp = 1/a^2
        SINUSOIDAL_CONE   the cone over (cos v, sin v, amp cos(folds v)):
                 cylindrical equation z = amp rho cos(n theta)
        HELICOIDAL_CONE   the cone over the helix (cos v, sin v, amp v),
                 v running `turns` turns: S = (u cos v, u sin v, amp u v)
    """
    if kind == 'PARABOLIC_CONOID':
        # right conoid with axis Oy: rulings parallel to the xz-plane
        # joining the axis point (0, u, 0) to the parabola directrix
        # (extent, u, amp extent (1 - u^2)) in the plane x = extent
        u = np.linspace(-1.3, 1.3, res_u)
        v = np.linspace(0.0, extent, res_v + 1)
        P = np.empty((res_u, res_v + 1, 3))
        for j, vv in enumerate(v):
            P[:, j, 0] = vv
            P[:, j, 1] = u
            P[:, j, 2] = amp * vv * (1.0 - u * u)
        return _mesh_grid(P)
    if kind == 'SINUSOIDAL_CONE':
        n = max(1, int(folds))
        v = np.linspace(0.0, _TWO_PI, res_u, endpoint=False)
        return _cone_over(res_v, extent,
                          np.cos(v), np.sin(v), amp * np.cos(n * v))
    if kind == 'HELICOIDAL_CONE':
        v = np.linspace(0.0, _TWO_PI * max(0.25, turns), res_u)
        return _cone_over(res_v, extent,
                          np.cos(v), np.sin(v), amp * v, wrap=False)
    if kind == 'ZINDLER':
        # Zindler's conoid, z = a tan(n theta) -- cartesian
        # z(x^2 - y^2) = 2 a x y for the classical n = 2.  Its height is
        # UNBOUNDED at the 2n asymptotes theta = (2k+1) pi / (2n), so
        # sampling theta uniformly (as every other conoid here does)
        # would spend the whole mesh on a pair of spikes and still clip
        # them.  Sample the HEIGHT uniformly instead and invert the
        # tangent: exact, bounded, and it grades the angular samples
        # toward the asymptotes automatically, which is where the sheet
        # actually turns.
        #
        # Each of the 2n branches is its own open patch.  They meet only
        # along Oz, which is not a defect: Oz is the conoid's double
        # line, and the directrix -- a tangentoid crown -- really does
        # have 2n separate branches.
        n = max(1, int(folds))
        cap = 2.0 * extent                     # vertical reach
        a = max(1e-6, abs(amp))
        h = np.linspace(-cap, cap, res_u)
        vv = np.linspace(-extent, extent, res_v + 1)
        verts, faces = [], []
        for k in range(2 * n):
            u = (np.arctan(h / a) + k * math.pi) / n
            cu, su = np.cos(u), np.sin(u)
            P = np.empty((res_u, res_v + 1, 3))
            for j, t in enumerate(vv):
                P[:, j, 0] = t * cu
                P[:, j, 1] = t * su
                P[:, j, 2] = h
            V, F = _mesh_grid(P)
            off = len(verts)
            verts.extend(V)
            faces.extend([[i + off for i in f] for f in F])
        return verts, faces
    if kind == 'WHITNEY':
        s = np.linspace(-extent, extent, res_u)
        v = np.linspace(-extent, extent, res_v + 1)
        P = np.empty((res_u, res_v + 1, 3))
        for j, vv in enumerate(v):
            P[:, j, 0] = s * vv
            P[:, j, 1] = s
            P[:, j, 2] = vv * vv
        return _mesh_grid(P)
    u = np.linspace(0.0, _TWO_PI, res_u, endpoint=False)
    if kind == 'PLUCKER':
        h = amp * np.sin(2.0 * u)
    elif kind == 'NFOLD':
        h = amp * np.sin(max(1, int(folds)) * u)
    else:  # WALLIS
        h = amp * np.sqrt(np.maximum(
            wallis_a ** 2 - wallis_b ** 2 * np.cos(u) ** 2, 0.0))
    cu, su = np.cos(u), np.sin(u)
    v = np.linspace(-extent, extent, res_v + 1)
    P = np.empty((res_u, res_v + 1, 3))
    for j, vv in enumerate(v):
        P[:, j, 0] = vv * cu
        P[:, j, 1] = vv * su
        P[:, j, 2] = h
    return _mesh_grid(P, wrap_u=True)


def rulings_conoid(kind='PLUCKER', amp=0.5, folds=2, wallis_a=1.0,
                   wallis_b=0.6, extent=1.0, n=48, turns=2.0):
    if kind == 'PARABOLIC_CONOID':
        segs = []
        for i in range(n + 1):
            y = -1.3 + 2.6 * i / n
            segs.append(((0.0, y, 0.0),
                         (extent, y, amp * extent * (1.0 - y * y))))
        return segs
    if kind == 'SINUSOIDAL_CONE':
        m = max(1, int(folds))
        segs = []
        for i in range(n):
            v = _TWO_PI * i / n
            segs.append(((0.0, 0.0, 0.0),
                         (extent * math.cos(v), extent * math.sin(v),
                          extent * amp * math.cos(m * v))))
        return segs
    if kind == 'HELICOIDAL_CONE':
        span = _TWO_PI * max(0.25, turns)
        segs = []
        for i in range(n + 1):
            v = span * i / n
            segs.append(((0.0, 0.0, 0.0),
                         (extent * math.cos(v), extent * math.sin(v),
                          extent * amp * v)))
        return segs
    if kind == 'ZINDLER':
        # same height-first sampling as the surface, so the rods land on
        # the rulings the mesh actually shows
        m = max(1, int(folds))
        cap, a = 2.0 * extent, max(1e-6, abs(amp))
        segs = []
        for k in range(2 * m):
            for i in range(n):
                hgt = -cap + 2.0 * cap * i / max(1, n - 1)
                u = (math.atan(hgt / a) + k * math.pi) / m
                cu, su = math.cos(u), math.sin(u)
                segs.append(((-extent * cu, -extent * su, hgt),
                             (extent * cu, extent * su, hgt)))
        return segs
    if kind == 'WHITNEY':
        # S = (u v, u, v^2) is straight along u at each fixed v: the
        # ruling through (0, 0, v^2) in the direction (v, 1, 0)
        segs = []
        for i in range(n + 1):
            w = -extent + 2.0 * extent * i / n
            segs.append(((-extent * w, -extent, w * w),
                         (extent * w, extent, w * w)))
        return segs
    segs = []
    for i in range(n):
        u = _TWO_PI * i / n
        if kind == 'PLUCKER':
            h = amp * math.sin(2.0 * u)
        elif kind == 'NFOLD':
            h = amp * math.sin(max(1, int(folds)) * u)
        else:
            h = amp * math.sqrt(max(
                wallis_a ** 2 - wallis_b ** 2 * math.cos(u) ** 2, 0.0))
        cu, su = math.cos(u), math.sin(u)
        segs.append(((-extent * cu, -extent * su, h),
                     (extent * cu, extent * su, h)))
    return segs


# --------------------------------------------------------------------
# 5. tangent developable of a helix
# --------------------------------------------------------------------

def build_tangent_developable(radius=1.0, pitch=0.35, turns=2.5,
                              v_extent=1.0, v_min=0.04,
                              res_u=200, res_v=16):
    """Tangent developable T(u,v)=c(u)+v c'(u) of the circular helix
    c(u)=(R cos u, R sin u, pitch*u).  Developable (unrolls flat); the
    helix itself (v=0) is the cuspidal edge of regression, so v runs
    from v_min > 0 outward to avoid the fold."""
    u = np.linspace(0.0, _TWO_PI * turns, res_u)
    cx, cy, cz = radius * np.cos(u), radius * np.sin(u), pitch * u
    dx, dy, dz = -radius * np.sin(u), radius * np.cos(u), \
        np.full_like(u, pitch)
    v = np.linspace(v_min, v_extent, res_v + 1)
    P = np.empty((res_u, res_v + 1, 3))
    for j, vv in enumerate(v):
        P[:, j, 0] = cx + vv * dx
        P[:, j, 1] = cy + vv * dy
        P[:, j, 2] = cz + vv * dz
    return _mesh_grid(P)


def rulings_tangent_developable(radius=1.0, pitch=0.35, turns=2.5,
                                v_extent=1.0, v_min=0.04, n=64):
    u = np.linspace(0.0, _TWO_PI * turns, n)
    segs = []
    for i in range(n):
        c = (radius * math.cos(u[i]), radius * math.sin(u[i]),
             pitch * u[i])
        d = (-radius * math.sin(u[i]), radius * math.cos(u[i]), pitch)
        segs.append(((c[0] + v_min * d[0], c[1] + v_min * d[1],
                      c[2] + v_min * d[2]),
                     (c[0] + v_extent * d[0], c[1] + v_extent * d[1],
                      c[2] + v_extent * d[2])))
    return segs


# --------------------------------------------------------------------
# 6. right / oblique helicoid
# --------------------------------------------------------------------

def build_helicoid(radius=1.0, pitch=0.4, turns=2.0, slope=0.0,
                   inner=0.0, res_u=200, res_v=12):
    """Radial ruling screwing up the z-axis:
        S(u,v) = (v cos u, v sin u, pitch*u + slope*v),  v in [inner,R].
    slope 0 is the right (minimal) helicoid; slope != 0 tilts the
    rulings off the horizontal (an oblique helicoid)."""
    u = np.linspace(0.0, _TWO_PI * turns, res_u)
    cu, su = np.cos(u), np.sin(u)
    zc = pitch * u
    v = np.linspace(inner, radius, res_v + 1)
    P = np.empty((res_u, res_v + 1, 3))
    for j, vv in enumerate(v):
        P[:, j, 0] = vv * cu
        P[:, j, 1] = vv * su
        P[:, j, 2] = zc + slope * vv
    return _mesh_grid(P)


def rulings_helicoid(radius=1.0, pitch=0.4, turns=2.0, slope=0.0,
                     inner=0.0, n=64):
    u = np.linspace(0.0, _TWO_PI * turns, n)
    segs = []
    for i in range(n):
        cu, su = math.cos(u[i]), math.sin(u[i])
        z = pitch * u[i]
        segs.append(((inner * cu, inner * su, z + slope * inner),
                     (radius * cu, radius * su, z + slope * radius)))
    return segs


# --------------------------------------------------------------------
# 7. n-half-twist strip (Mobius band for odd n)
# --------------------------------------------------------------------

def build_twist_strip(radius=1.0, width=0.4, half_twists=1,
                      res_u=200, res_v=6):
    """n-half-twist ruled band around a circle of radius R, strip
    half-width w.  b(u)=(R cos u,R sin u,0);
    d(u)=(cos(nu/2)cos u, cos(nu/2)sin u, sin(nu/2)).  Odd n -> Mobius
    (one-sided); even n -> orientable twisted annulus.  Needs w < R."""
    n = half_twists
    u = np.linspace(0.0, _TWO_PI, res_u + 1)   # include the seam
    v = np.linspace(-width, width, res_v + 1)
    cu, su = np.cos(u), np.sin(u)
    ch, sh = np.cos(n * u / 2.0), np.sin(n * u / 2.0)
    P = np.empty((res_u + 1, res_v + 1, 3))
    for j, vv in enumerate(v):
        rr = radius + vv * ch
        P[:, j, 0] = rr * cu
        P[:, j, 1] = rr * su
        P[:, j, 2] = vv * sh
    return _mesh_grid(P)


def rulings_twist_strip(radius=1.0, width=0.4, half_twists=1, n=64):
    hn = half_twists
    segs = []
    for i in range(n):
        u = _TWO_PI * i / n
        cu, su = math.cos(u), math.sin(u)
        ch, sh = math.cos(hn * u / 2.0), math.sin(hn * u / 2.0)
        p0 = ((radius - width * ch) * cu, (radius - width * ch) * su,
              -width * sh)
        p1 = ((radius + width * ch) * cu, (radius + width * ch) * su,
              width * sh)
        segs.append((p0, p1))
    return segs


# --------------------------------------------------------------------
# 8. doubly-ruled hyperbolic paraboloid (hypar)
# --------------------------------------------------------------------

def build_hypar(a=1.0, b=1.0, c=1.0, extent=1.0, res=48,
                corners=None):
    """Doubly-ruled saddle.  With corners=None: z = c((x/a)^2-(y/b)^2)
    over [-extent,extent]^2.  With corners = (P00,P10,P01,P11) (four
    3D points in general position): the bilinear Coons patch
        S(s,t) = (1-s)(1-t)P00 + s(1-t)P10 + (1-s)t P01 + s t P11,
    whose s- and t-edges are straight -- the surface of four skew
    points is always a hypar."""
    n = max(2, int(res))
    if corners is not None:
        P00, P10, P01, P11 = (np.asarray(p, dtype=float)
                              for p in corners)
        s = np.linspace(0.0, 1.0, n + 1)
        t = np.linspace(0.0, 1.0, n + 1)
        P = np.empty((n + 1, n + 1, 3))
        for i, ss in enumerate(s):
            for j, tt in enumerate(t):
                P[i, j] = ((1 - ss) * (1 - tt) * P00
                           + ss * (1 - tt) * P10
                           + (1 - ss) * tt * P01
                           + ss * tt * P11)
        return _mesh_grid(P)
    x = np.linspace(-extent, extent, n + 1)
    y = np.linspace(-extent, extent, n + 1)
    P = np.empty((n + 1, n + 1, 3))
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            P[i, j] = (xx, yy,
                       c * ((xx / a) ** 2 - (yy / b) ** 2))
    return _mesh_grid(P)


def rulings_hypar(a=1.0, b=1.0, c=1.0, extent=1.0, corners=None,
                  n=16, family='BOTH'):
    """Ruling segments of the hypar.  RIGHT keeps one family (the lines
    x/a + y/b = const, or the patch's constant-s lines), LEFT the other,
    BOTH interleaves them.  Every segment of a family runs the same way
    (increasing x, or increasing t / s), which the woven output needs."""
    right = family in ('RIGHT', 'BOTH')
    left = family in ('LEFT', 'BOTH')
    segs = []
    if corners is not None:
        P00, P10, P01, P11 = (np.asarray(p, dtype=float)
                              for p in corners)
        for i in range(n + 1):
            s = i / n
            if right:
                e0 = (1 - s) * P00 + s * P10
                e1 = (1 - s) * P01 + s * P11
                segs.append((tuple(e0), tuple(e1)))    # t-rulings
            if left:
                f0 = (1 - s) * P00 + s * P01
                f1 = (1 - s) * P10 + s * P11
                segs.append((tuple(f0), tuple(f1)))    # s-rulings
        return segs
    # z = c((x/a)^2 - (y/b)^2) factors as c*u*v in the skew coordinates
    #     u = x/a + y/b,      v = x/a - y/b,
    # so u = const and v = const are BOTH straight lines lying on it --
    # that is what "doubly ruled" means here, and neither family runs
    # along x or y.  On u = k the height is c*k*(2x/a - k), linear in x;
    # likewise on v = k.  (The earlier code swept y and joined
    # (-extent, y) to (+extent, y) at a single height: a horizontal
    # chord that meets the surface only at its two ends, and one whose
    # family sweeps a parabolic cylinder rather than the saddle.)
    #
    # Each line is clipped to the square domain build_hypar meshes, so
    # the rods and the surface cover the same region.  Lines near the
    # two extreme k are genuinely short -- a square-domain saddle's
    # rulings cross it diagonally and taper to nothing at the corners.
    ka, kb = extent / a, extent / b
    kmax = ka + kb
    for i in range(n + 1):
        k = -kmax + 2.0 * kmax * i / n
        # y in [-extent, extent]  <=>  x/a in [k - kb, k + kb]
        x0 = a * max(k - kb, -ka)
        x1 = a * min(k + kb, ka)
        if x1 - x0 <= 1e-12:
            continue                       # this line misses the square
        for sign in (1.0, -1.0):           # u = k, then v = k
            if not (right if sign > 0 else left):
                continue
            ends = []
            for xx in (x0, x1):
                yy = sign * b * (k - xx / a)
                ends.append((xx, yy,
                             c * ((xx / a) ** 2 - (yy / b) ** 2)))
            segs.append((ends[0], ends[1]))
    return segs


# --------------------------------------------------------------------
# mode dispatch (pure)
# --------------------------------------------------------------------

def developable_determinant(bx, by, bz, dx, dy, dz, u=None):
    """det[b'(u), d(u), d'(u)] sampled numerically along a ruled
    surface; ~0 everywhere <=> developable (unrolls flat).  Pass the
    parameter samples `u` so the derivatives use the real step."""
    g = (lambda a: np.gradient(a, u)) if u is not None else np.gradient
    bp = np.stack([g(bx), g(by), g(bz)], axis=1)
    d = np.stack([dx, dy, dz], axis=1)
    dp = np.stack([g(dx), g(dy), g(dz)], axis=1)
    return np.array([np.linalg.det(np.stack([bp[i], d[i], dp[i]]))
                     for i in range(len(bx))])


_MODES = [
    ('HYPERBOLOID', "Hyperboloid",
     "Hyperboloid of one sheet from straight rulings between two "
     "coaxial circles; twist sets the waist"),
    ('HELICAL_CONE', "Compound Helical Cone",
     "Solomonic column: a spirally-fluted cone wound helically "
     "toward its apex"),
    ('SPIRAL', "Spiral Ruled",
     "Farris' spiral ruled surface: a log-spiral or rosette base "
     "swept with a tangent+vertical ruling"),
    ('CONOID', "Conoid",
     "Right conoids: Plucker's cylindroid, n-fold, Wallis conical "
     "edge, and the Whitney umbrella"),
    ('TANGENT_DEV', "Tangent Developable",
     "The developable (flat-unrollable) surface swept by the "
     "tangents of a helix"),
    ('HELICOID', "Helicoid",
     "Right (minimal) or oblique helicoid: a radial ruling "
     "screwing up the axis"),
    ('TWIST_STRIP', "Twisted Strip",
     "n-half-twist ruled band; odd n is a one-sided Mobius band"),
    ('HYPAR', "Hyperbolic Paraboloid",
     "Doubly-ruled saddle, as z=c((x/a)^2-(y/b)^2) or a bilinear "
     "patch of four skew corner points"),
    ('KNOT_SPAN', "Concentric Toroidal Knots",
     "Straight rulings strung between two concentric (p, q) torus "
     "knots; outer q=0 degenerates the outer curve to a circle"),
    ('GAUDI', "Gaudi's Surface",
     "Sinusoidal conoid z = k x sin(y/a); Gaudi roofed the Sagrada "
     "Familia escoles with it, a warped surface built from straight "
     "timber"),
    ('GUIMARD', "Guimard's Surface",
     "The lines joining a point moving sinusoidally along a line to a "
     "point on a doubled sinusoid round a circle; Art Nouveau ironwork"),
    ('MILK_CARTON', "Milk Carton",
     "The berlingot: elliptical sections whose axes trade places with "
     "height, closing to two perpendicular segments at top and bottom"),
    ('RULED_CUBIC', "Skew Ruled Cubic",
     "A conic, a line meeting its plane, and a homography between them; "
     "the joining lines sweep a cubic surface"),
    ('CONSTANT_SLOPE', "Surface of Constant Slope",
     "Monge's sandpile: rise from a closed base curve along its normal "
     "at a fixed angle, so every tangent plane leans the same way"),
]

#: the Output a mode starts from when it is not rods.  The compound
#: helical cone is the one: it is not straight-ruled, so its rods draw
#: arrises rather than rulings, and it reads as a column only when filled.
_DEFAULT_OUTPUT = {'HELICAL_CONE': 'SURFACE'}

#: modes with two ruling families that cross on one surface, and so a
#: Ruling Family choice: the two doubly-ruled quadrics, and the spiral and
#: the knot span, whose left families are laid on their surfaces as
#: curves (`left_rulings_spiral`, `left_rulings_knot_span`)
_TWO_FAMILY = {'HYPERBOLOID', 'HYPAR', 'KNOT_SPAN', 'SPIRAL'}

#: every other straight-ruled mode, woven as a basket: its rulings with
#: the curves running along the surface (`weft_weave`, `_weft_charts`)
_WEFT = {'CONOID', 'TANGENT_DEV', 'HELICOID', 'TWIST_STRIP', 'GAUDI',
         'GUIMARD', 'MILK_CARTON', 'RULED_CUBIC', 'CONSTANT_SLOPE'}

#: modes with the woven-ribbon output: every one but the compound helical
#: cone, which has no rulings.  A mode joins by answering `_weave_input`.
_WOVEN = _TWO_FAMILY | _WEFT


def default_output(mode):
    """The Output a mode starts from: rods, unless `_DEFAULT_OUTPUT`
    says otherwise."""
    return _DEFAULT_OUTPUT.get(mode, 'RODS')


def output_after_mode_change(output, old_mode, new_mode):
    """The Output to show once the surface changes from `old_mode` to
    `new_mode`.  An Output still at the old surface's default moves to the
    new one's; one chosen by hand is kept, so switching surfaces never
    throws a deliberate choice away."""
    if old_mode != new_mode and output == default_output(old_mode):
        return default_output(new_mode)
    return output


def effective_output(op):
    """The output actually built.  Woven ribbons need two crossing ruling
    families, so on a mode with one they fall back to rods."""
    if op.output == 'RIBBONS' and op.mode not in _WOVEN:
        return 'RODS'
    return op.output


#: the five surfaces above, which share one builder
_NAMED_MODES = ('GAUDI', 'GUIMARD', 'MILK_CARTON', 'RULED_CUBIC',
                'CONSTANT_SLOPE')

_NAMED_LABELS = [
    ('GAUDI', "Gaudi's Surface",
     "Sinusoidal conoid z = k x sin(y/a); Gaudi roofed the Sagrada "
     "Familia escoles with it, a warped surface built from straight "
     "timber"),
    ('GUIMARD', "Guimard's Surface",
     "The lines joining a point moving sinusoidally along a line to a "
     "point on a doubled sinusoid round a circle; Art Nouveau ironwork"),
    ('MILK_CARTON', "Milk Carton",
     "The berlingot: elliptical sections whose axes trade places with "
     "height, closing to two perpendicular segments at top and bottom"),
    ('RULED_CUBIC', "Skew Ruled Cubic",
     "A conic, a line meeting its plane, and a homography between them; "
     "the joining lines sweep a cubic surface"),
    ('CONSTANT_SLOPE', "Surface of Constant Slope",
     "Monge's sandpile: rise from a closed base curve along its normal "
     "at a fixed angle, so every tangent plane leans the same way"),
]

_CONOID_KINDS = [
    ('PLUCKER', "Plucker Cylindroid", "h = amp sin(2u): two leaves"),
    ('NFOLD', "n-Fold Conoid", "h = amp sin(folds * u)"),
    ('WALLIS', "Wallis Conical Edge",
     "h = amp sqrt(a^2 - b^2 cos^2 u)"),
    ('ZINDLER', "Zindler Conoid",
     "h = a tan(folds * u): the cubic ruled surface "
     "z(x^2 - y^2) = 2 a x y, with Oz as its double line"),
    ('WHITNEY', "Whitney Umbrella",
     "S = (uv, u, v^2): a pinch-point ruled surface"),
    ('PARABOLIC_CONOID', "Parabolic Conoid",
     "The conoid whose curved directrix is a parabola: the roof-shell "
     "surface a^2 z = x(b^2 - y^2), rulings joining the axis line to "
     "the parabola.  When the parabola's axis is instead parallel to "
     "the conoid's axis it degenerates to the Whitney umbrella"),
    ('SINUSOIDAL_CONE', "Sinusoidal Cone",
     "The cone with apex at the origin over the closed sine-wave "
     "crown: cylindrical equation z = k rho cos(n theta).  Distinct "
     "from Plucker's conoid z = a cos(n theta), whose rulings do not "
     "pass through one point"),
    ('HELICOIDAL_CONE', "Helicoidal Cone",
     "The cone joining a fixed apex to a circular helix: rulings "
     "(u cos v, u sin v, k u v).  A cone, not a helicoid -- every "
     "ruling passes through the apex"),
]

# modes that are genuinely straight-ruled -> rods available
_RULED = ({'HYPERBOLOID', 'SPIRAL', 'CONOID', 'TANGENT_DEV', 'HELICOID',
           'TWIST_STRIP', 'HYPAR', 'KNOT_SPAN', 'HELICAL_CONE'}
          | set(_NAMED_MODES))


def _twist_deg(op):
    """The hyperboloid's twist in degrees, as its builders take it."""
    return math.degrees(op.twist_angle)


def _shape_controls(op):
    """The shape controls the panel lists for the current surface, in
    order, as (property, label or None for its own, editable).

    Every surface lists the same way: what sets its size, then its
    shape, then how far its rulings reach.  The concentric toroidal knots
    list each knot as p, q, scale, tube, height -- the outer knot's p
    greyed out while its q is 0, when it is a circle wound the inner p
    times -- and the twist that pairs them last, as the hyperboloid does.
    The named surfaces share a few properties whose meaning differs from
    one to the next, so those are shown under what they do there."""
    m = op.mode
    out = []

    def add(key, text=None, on=True):
        out.append((key, text, on))

    if m == 'HYPERBOLOID':
        for k in ('radius', 'height', 'twist_angle'):
            add(k)
    elif m == 'HELICAL_CONE':
        for k in ('radius', 'cone_height', 'flutes', 'flute_depth',
                  'cone_twist', 'taper', 'orbit_amp', 'orbit_turns'):
            add(k)
    elif m == 'SPIRAL':
        for k in ('tightness', 'petals', 'petal_amp', 'turns', 'slope',
                  'v_extent'):
            add(k)
    elif m == 'CONOID':
        add('conoid_kind')
        add('amp')
        if op.conoid_kind in ('NFOLD', 'ZINDLER', 'SINUSOIDAL_CONE'):
            add('folds')
        elif op.conoid_kind == 'WALLIS':
            add('wallis_a')
            add('wallis_b')
        elif op.conoid_kind == 'HELICOIDAL_CONE':
            add('turns')
        add('v_extent')
    elif m == 'TANGENT_DEV':
        for k in ('radius', 'pitch', 'turns', 'v_min', 'v_extent'):
            add(k)
    elif m == 'HELICOID':
        for k in ('radius', 'inner', 'pitch', 'turns', 'slope'):
            add(k)
    elif m == 'TWIST_STRIP':
        for k in ('radius', 'width', 'half_twists'):
            add(k)
    elif m == 'KNOT_SPAN':
        for k in ('knot_p', 'knot_q', 'knot_scale', 'knot_tube',
                  'knot_inner_height', 'knot_inner_lift', 'knot_rotation'):
            add(k)
        knotted = op.knot_outer_q > 0
        add('knot_outer_p', None, knotted)
        add('knot_outer_q')
        if knotted:
            for k in ('knot_outer_scale', 'knot_outer_tube',
                      'knot_outer_height'):
                add(k)
        else:
            add('knot_circle_radius')
        add('knot_twist')
    elif m == 'GAUDI':
        add('amp')
        add('wallis_a', "Length")
        add('v_extent', "Half Width")
    elif m == 'GUIMARD':
        add('wallis_a', "Line Swing")
        add('v_extent', "Circle Radius")
        add('amp', "Wave Height")
    elif m == 'MILK_CARTON':
        add('amp', "Width")
        add('v_extent', "Half Height")
    elif m == 'RULED_CUBIC':
        add('v_extent', "Height")
    elif m == 'CONSTANT_SLOPE':
        add('folds', "Lobes")
        add('wallis_b', "Lobe Depth")
        add('wallis_a', "Steepness")
        add('v_extent', "Rise")
    else:  # HYPAR
        add('use_corners')
        if op.use_corners:
            for k in ('p00', 'p10', 'p01', 'p11'):
                add(k)
        else:
            for k in ('hy_a', 'hy_b', 'hy_c', 'v_extent'):
                add(k)
    return out


def _build_surface(op):
    """(verts, faces, name) for the operator's current mode."""
    m = op.mode
    if m == 'HYPERBOLOID':
        vf = build_hyperboloid(op.radius, op.height, _twist_deg(op),
                               op.res_u, op.res_v)
        return (*vf, "Hyperboloid")
    if m == 'HELICAL_CONE':
        vf = build_helical_cone(op.radius, op.cone_height, op.flutes,
                                op.flute_depth, op.cone_twist,
                                op.taper, op.orbit_amp,
                                op.orbit_turns, op.res_u, op.res_v)
        return (*vf, "Compound Helical Cone")
    if m == 'SPIRAL':
        vf = build_spiral_ruled(op.tightness, op.slope, op.turns,
                                op.petals, op.petal_amp, op.v_extent,
                                op.res_u, op.res_v)
        return (*vf, "Spiral Ruled Surface")
    if m == 'CONOID':
        vf = build_conoid(op.conoid_kind, op.amp, op.folds,
                          op.wallis_a, op.wallis_b, op.v_extent,
                          op.res_u, op.res_v, turns=op.turns)
        return (*vf, dict((i[0], i[1])
                          for i in _CONOID_KINDS)[op.conoid_kind])
    if m in _NAMED_MODES:
        vf = build_named_ruled(m, op.amp, op.v_extent, op.folds,
                               op.wallis_a, op.wallis_b,
                               op.res_u, op.res_v)
        return (*vf, dict((i[0], i[1]) for i in _NAMED_LABELS)[m])
    if m == 'TANGENT_DEV':
        vf = build_tangent_developable(op.radius, op.pitch, op.turns,
                                       op.v_extent, op.v_min,
                                       op.res_u, op.res_v)
        return (*vf, "Tangent Developable")
    if m == 'HELICOID':
        vf = build_helicoid(op.radius, op.pitch, op.turns, op.slope,
                            op.inner, op.res_u, op.res_v)
        return (*vf, "Helicoid")
    if m == 'TWIST_STRIP':
        vf = build_twist_strip(op.radius, op.width, op.half_twists,
                               op.res_u, op.res_v)
        return (*vf, "Twisted Strip")
    if m == 'KNOT_SPAN':
        vf = build_knot_span(op.knot_p, op.knot_q, op.knot_scale,
                             op.knot_tube, op.knot_inner_height,
                             op.knot_inner_lift, op.knot_rotation,
                             op.knot_outer_p, op.knot_outer_q,
                             op.knot_outer_scale, op.knot_outer_tube,
                             op.knot_outer_height, op.knot_circle_radius,
                             op.res_u, op.res_v, shift=op.knot_twist)
        return (*vf, "Concentric Toroidal Knots")
    # HYPAR
    corners = None
    if op.use_corners:
        corners = (op.p00, op.p10, op.p01, op.p11)
    vf = build_hypar(op.hy_a, op.hy_b, op.hy_c, op.v_extent,
                     op.res_u, corners)
    return (*vf, "Hyperbolic Paraboloid")


def _build_rulings(op, n=None):
    """Ruling segments for the current mode (rods mode)."""
    m = op.mode
    if n is None:
        n = op.knot_rods if m == 'KNOT_SPAN' else op.n_rods
    if m == 'HYPERBOLOID':
        return rulings_hyperboloid(op.radius, op.height, _twist_deg(op),
                                   op.family, n)
    if m == 'HELICAL_CONE':
        return arrises_helical_cone(
            op.radius, op.cone_height, op.flutes, op.flute_depth,
            op.cone_twist, op.taper, op.orbit_amp, op.orbit_turns,
            max(8, n // max(1, int(op.flutes))))
    if m == 'SPIRAL':
        return rulings_spiral(op.tightness, op.slope, op.turns,
                              op.petals, op.petal_amp, op.v_extent, n,
                              op.family)
    if m == 'CONOID':
        return rulings_conoid(op.conoid_kind, op.amp, op.folds,
                              op.wallis_a, op.wallis_b, op.v_extent, n,
                              turns=op.turns)
    if m in _NAMED_MODES:
        return rulings_named_ruled(m, op.amp, op.v_extent, op.folds,
                                   op.wallis_a, op.wallis_b, n)
    if m == 'TANGENT_DEV':
        return rulings_tangent_developable(op.radius, op.pitch,
                                           op.turns, op.v_extent,
                                           op.v_min, n)
    if m == 'HELICOID':
        return rulings_helicoid(op.radius, op.pitch, op.turns,
                                op.slope, op.inner, n)
    if m == 'TWIST_STRIP':
        return rulings_twist_strip(op.radius, op.width,
                                   op.half_twists, n)
    if m == 'KNOT_SPAN':
        return rulings_knot_span(op.knot_p, op.knot_q, op.knot_scale,
                                 op.knot_tube, op.knot_inner_height,
                                 op.knot_inner_lift, op.knot_rotation,
                                 op.knot_outer_p, op.knot_outer_q,
                                 op.knot_outer_scale, op.knot_outer_tube,
                                 op.knot_outer_height,
                                 op.knot_circle_radius, n, op.family,
                                 op.knot_twist)
    if m == 'HYPAR':
        corners = (op.p00, op.p10, op.p01, op.p11) \
            if op.use_corners else None
        return rulings_hypar(op.hy_a, op.hy_b, op.hy_c, op.v_extent,
                             corners, n, op.family)
    return []


def _build_curves(op, n=None):
    """Curved rulings for the current mode, as polylines.  Only the spiral
    and the knot span have any: their left families lie on the surface
    but bend there (`left_rulings_spiral`, `left_rulings_knot_span`).
    Every other ruling is straight and comes from `_build_rulings`."""
    if op.family not in ('LEFT', 'BOTH'):
        return []
    if op.mode == 'SPIRAL':
        return left_rulings_spiral(op.tightness, op.slope, op.turns,
                                   op.petals, op.petal_amp, op.v_extent,
                                   op.n_rods if n is None else n)
    if op.mode != 'KNOT_SPAN':
        return []
    return left_rulings_knot_span(
        op.knot_p, op.knot_q, op.knot_scale, op.knot_tube,
        op.knot_inner_height, op.knot_inner_lift, op.knot_rotation,
        op.knot_outer_p, op.knot_outer_q, op.knot_outer_scale,
        op.knot_outer_tube, op.knot_outer_height, op.knot_circle_radius,
        op.knot_rods if n is None else n, op.knot_twist)


def _ruling_families(op, n=None):
    """The two ruling families of a mode that has two, as a pair of
    strand lists -- segments, or polylines for the knot span's curved
    left family -- each family oriented consistently; None for a mode
    with only one family."""
    if op.mode == 'KNOT_SPAN':
        right, left, _x, _g = _knot_span_weave(op, n)
        return right, left
    if op.mode == 'SPIRAL':
        right, left, _x, _g = _spiral_weave(op, n)
        return right, left
    if op.mode in _WEFT:
        right, left, _x, _g = weft_weave(_weft_charts(op, n))
        return right, left
    if n is None:
        n = op.n_rods
    if op.mode == 'HYPERBOLOID':
        # both families run bottom circle -> top circle
        return (rulings_hyperboloid(op.radius, op.height, _twist_deg(op),
                                    'RIGHT', n),
                rulings_hyperboloid(op.radius, op.height, _twist_deg(op),
                                    'LEFT', n))
    if op.mode == 'HYPAR':
        corners = (op.p00, op.p10, op.p01, op.p11) \
            if op.use_corners else None
        return tuple(rulings_hypar(op.hy_a, op.hy_b, op.hy_c,
                                   op.v_extent, corners, n, fam)
                     for fam in ('RIGHT', 'LEFT'))
    return None


def _knot_span_weave(op, n=None, overhang=0.0):
    return knot_span_weave(
        op.knot_p, op.knot_q, op.knot_scale, op.knot_tube,
        op.knot_inner_height, op.knot_inner_lift, op.knot_rotation,
        op.knot_outer_p, op.knot_outer_q, op.knot_outer_scale,
        op.knot_outer_tube, op.knot_outer_height, op.knot_circle_radius,
        op.knot_rods if n is None else n, op.knot_twist, overhang)


def _spiral_weave(op, n=None, overhang=0.0):
    return spiral_weave(op.tightness, op.slope, op.turns, op.petals,
                        op.petal_amp, op.v_extent,
                        op.n_rods if n is None else n, overhang)


def _weave_input(op):
    """What the woven output weaves: (family a, family b, options) for
    `weave_rulings`.  Straight families go in with their overhang and the
    weaver finds their crossings.  The spiral and the knot span, whose
    left families are curved, supply their exact crossings and neighbour
    spacing (`spiral_weave`, `knot_span_weave`), and size their ribbons
    cell by cell: the knot span's lattice is open in the outer flares and
    crowded by the inner knot, and the spiral's widens as it winds out,
    so one width for every ribbon would leave the open cells nearly
    empty."""
    if op.mode in ('KNOT_SPAN', 'SPIRAL'):
        weave = _knot_span_weave if op.mode == 'KNOT_SPAN' \
            else _spiral_weave
        fa, fb, crossings, gap = weave(op, overhang=op.ribbon_overhang)
        return fa, fb, dict(crossings=crossings, gap=gap,
                            local_width=True)
    if op.mode in _WEFT:
        fa, fb, crossings, gap = weft_weave(_weft_charts(op),
                                            op.ribbon_overhang)
        return fa, fb, dict(crossings=crossings, gap=gap,
                            local_width=True)
    fa, fb = extend_families(*_ruling_families(op), op.ribbon_overhang)
    return fa, fb, {}


def _weave_gap_used(op):
    """Whether the current surface's rulings run together somewhere, so
    that woven ribbons stop short of it by Weave Gap."""
    m = op.mode
    if m == 'CONOID':
        return op.conoid_kind != 'PARABOLIC_CONOID'
    if m == 'HELICOID':
        return abs(op.pitch) < 1e-9
    return m in ('TANGENT_DEV', 'GUIMARD', 'MILK_CARTON')


def _grid_surface(fn):
    """S(u, v) from fn(u, v) on equal-shaped float arrays."""
    def S(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float),
                                   np.asarray(v, dtype=float))
        return fn(u, v)
    return S


def _weft_charts(op, n=None):
    """The pieces of surface `weft_weave` weaves for a mode in `_WEFT`,
    in coordinates where each ruling is a line u = const.

    Most modes are one piece.  A conoid through its axis is woven as
    half-rulings from a gap by the axis outward: Plucker's, the Wallis
    edge and the even n-fold conoids cover each ruling twice (u and
    u + pi give the same line), so one ring of half-rulings is the whole
    surface; an odd n-fold conoid needs the ring on each side.  Zindler's
    conoid is one piece per branch, the Whitney umbrella one per side of
    its handle.  Cones stop short of the apex, the tangent developable
    short of its fold, Guimard's surface short of the segment its
    rulings pair off at, the milk carton short of both its seams."""
    n = op.n_rods if n is None else n
    m, e, g = op.mode, op.v_extent, op.weave_gap
    loop = (0.0, _TWO_PI)
    if m == 'CONOID':
        kind, amp = op.conoid_kind, op.amp
        f = max(1, int(op.folds))
        if kind == 'PARABOLIC_CONOID':
            S = _grid_surface(lambda u, v: np.stack(
                [v, u, amp * v * (1.0 - u * u)], -1))
            return [_weft_chart(S, n, (-1.3, 1.3), None, (0.0, e))]
        if kind == 'SINUSOIDAL_CONE':
            S = _grid_surface(lambda u, v: np.stack(
                [v * np.cos(u), v * np.sin(u), v * amp * np.cos(f * u)], -1))
            return [_weft_chart(S, n, loop, 'loop', (g * e, e),
                                (True, False))]
        if kind == 'HELICOIDAL_CONE':
            S = _grid_surface(lambda u, v: np.stack(
                [v * np.cos(u), v * np.sin(u), v * amp * u], -1))
            return [_weft_chart(S, n, (0.0, _TWO_PI * max(0.25, op.turns)),
                                None, (g * e, e), (True, False))]
        if kind == 'WHITNEY':
            S = _grid_surface(lambda w, t: np.stack([t * w, t, w * w], -1))
            return [_weft_chart(S, n, (-e, e), None, (g * e, e),
                                (True, False)),
                    _weft_chart(S, n, (-e, e), None, (-e, -g * e),
                                (False, True))]
        if kind == 'ZINDLER':
            cap, a = 2.0 * e, max(1e-6, abs(amp))
            charts = []
            for k in range(2 * f):
                def branch(h, v, k=k):
                    ang = (np.arctan(h / a) + k * math.pi) / f
                    return np.stack([v * np.cos(ang), v * np.sin(ang), h],
                                    -1)
                charts.append(_weft_chart(_grid_surface(branch), n,
                                          (-cap, cap), None, (g * e, e),
                                          (True, False)))
            return charts
        if kind == 'PLUCKER':
            def h(u):
                return amp * np.sin(2.0 * u)
        elif kind == 'NFOLD':
            def h(u):
                return amp * np.sin(f * u)
        else:  # WALLIS
            def h(u):
                return amp * np.sqrt(np.maximum(
                    op.wallis_a ** 2 - op.wallis_b ** 2 * np.cos(u) ** 2,
                    0.0))
        S = _grid_surface(lambda u, v: np.stack(
            [v * np.cos(u), v * np.sin(u), h(u)], -1))
        charts = [_weft_chart(S, n, loop, 'loop', (g * e, e), (True, False))]
        if kind == 'NFOLD' and f % 2:
            charts.append(_weft_chart(S, n, loop, 'loop', (-e, -g * e),
                                      (False, True)))
        return charts
    if m == 'TANGENT_DEV':
        R, p = op.radius, op.pitch
        S = _grid_surface(lambda u, v: np.stack(
            [R * np.cos(u) - v * R * np.sin(u),
             R * np.sin(u) + v * R * np.cos(u), p * (u + v)], -1))
        v0 = op.v_min + g * (e - op.v_min)
        return [_weft_chart(S, n, (0.0, _TWO_PI * op.turns), None, (v0, e),
                            (True, False))]
    if m == 'HELICOID':
        R, p, s = op.radius, op.pitch, op.slope
        S = _grid_surface(lambda u, v: np.stack(
            [v * np.cos(u), v * np.sin(u), p * u + s * v], -1))
        flat = abs(p) < 1e-9
        v0 = max(op.inner, g * R) if flat else op.inner
        return [_weft_chart(S, n, (0.0, _TWO_PI * op.turns), None, (v0, R),
                            (flat, False))]
    if m == 'TWIST_STRIP':
        R, w, hn = op.radius, op.width, int(op.half_twists)

        def band(u, v):
            rr = R + v * np.cos(hn * u / 2.0)
            return np.stack([rr * np.cos(u), rr * np.sin(u),
                             v * np.sin(hn * u / 2.0)], -1)
        return [_weft_chart(_grid_surface(band), n, loop,
                            'mobius' if hn % 2 else 'loop', (-w, w))]

    # the named surfaces: the lines joining their two curves
    def joined(u, v):
        A, B = named_ruled_curves(m, u.ravel(), op.amp, e, op.folds,
                                  op.wallis_a, op.wallis_b)
        P = A + v.ravel()[:, None] * (B - A)
        return P.reshape(u.shape + (3,))
    S = _grid_surface(joined)
    if m == 'GAUDI':
        a = op.wallis_a
        return [_weft_chart(S, n, (-math.pi * a, math.pi * a), None,
                            (0.0, 1.0))]
    if m == 'RULED_CUBIC':
        return [_weft_chart(S, n, (-1.45, 1.45), None, (0.0, 1.0))]
    if m == 'GUIMARD':
        return [_weft_chart(S, n, loop, 'loop', (g, 1.0), (True, False))]
    if m == 'MILK_CARTON':
        gg = min(g, 0.45)
        return [_weft_chart(S, n, loop, 'loop', (gg, 1.0 - gg),
                            (True, True))]
    return [_weft_chart(S, n, loop, 'loop', (0.0, 1.0))]    # CONSTANT_SLOPE


def _hypar_boundary(op, N):
    """The (closed) boundary loop of the hypar: the four corner edges of
    the bilinear patch, or the perimeter of the z=c((x/a)^2-(y/b)^2)
    square."""
    if op.use_corners:
        loop = [tuple(op.p00), tuple(op.p10), tuple(op.p11),
                tuple(op.p01)]
        return [(loop, True)]
    a, b, c, ext = op.hy_a, op.hy_b, op.hy_c, op.v_extent

    def z(x, y):
        return c * ((x / a) ** 2 - (y / b) ** 2)

    k = max(2, N // 4)
    xs = np.linspace(-ext, ext, k + 1)
    ys = np.linspace(-ext, ext, k + 1)
    loop = [(x, -ext, z(x, -ext)) for x in xs]
    loop += [(ext, y, z(ext, y)) for y in ys[1:]]
    loop += [(x, ext, z(x, ext)) for x in xs[::-1][1:]]
    loop += [(-ext, y, z(-ext, y)) for y in ys[::-1][1:-1]]
    return [(loop, True)]


def _boundary_loops(op):
    """The directrix / rail curves the rulings are strung between, as a
    list of (points, closed) polylines: the two boundary rails of the
    current mode (for KNOT_SPAN the inner and outer torus knots; for the
    hyperboloid the two coaxial circles, etc.).  These complete the
    'net' when a straight-ruled mode is drawn as rods or bare curves."""
    m = op.mode
    if m not in _RULED:
        return []
    # A rail is a smooth curve, not one point per stick: sample it far
    # more finely than the ruling count so the hoop reads as a circle.
    N = max(96, op.res_u)
    if m == 'HYPAR':
        return _hypar_boundary(op, N)
    if m == 'HELICAL_CONE':
        # the arrises all meet at the apex, so a "rail" through their
        # far ends would be a point; only the base circle is a curve
        segs = arrises_helical_cone(
            op.radius, op.cone_height, op.flutes, op.flute_depth,
            op.cone_twist, op.taper, op.orbit_amp, op.orbit_turns, 2)
        return [([s[0] for s in segs[::2]], True)]
    if m == 'HYPERBOLOID':
        # a single ruling family gives clean bottom/top rails
        segs = rulings_hyperboloid(op.radius, op.height, _twist_deg(op),
                                   'RIGHT', N)
    elif m == 'KNOT_SPAN':
        # likewise: with BOTH the two families' ends would interleave
        # and the rail would zigzag between them
        segs = rulings_knot_span(op.knot_p, op.knot_q, op.knot_scale,
                                 op.knot_tube, op.knot_inner_height,
                                 op.knot_inner_lift, op.knot_rotation,
                                 op.knot_outer_p, op.knot_outer_q,
                                 op.knot_outer_scale, op.knot_outer_tube,
                                 op.knot_outer_height,
                                 op.knot_circle_radius, N, 'RIGHT',
                                 op.knot_twist)
    elif m == 'SPIRAL':
        # the right family alone: its ends are the two rails
        segs = rulings_spiral(op.tightness, op.slope, op.turns, op.petals,
                              op.petal_amp, op.v_extent, N)
    else:
        segs = _build_rulings(op, N)
    if not segs:
        return []
    rail0 = [s[0] for s in segs]
    rail1 = [s[1] for s in segs]
    if m == 'CONOID' and op.conoid_kind in ('SINUSOIDAL_CONE',
                                            'HELICOIDAL_CONE'):
        # every ruling starts at the apex, so rail0 is a single point;
        # only the directrix (the sine-wave crown, or the helix, which
        # is open) is a curve
        return [(rail1, op.conoid_kind == 'SINUSOIDAL_CONE')]
    closed = (m in ('HYPERBOLOID', 'KNOT_SPAN', 'TWIST_STRIP')
              # Zindler's directrix has 2n separate branches running off
              # to infinity, so its rails are open like Whitney's --
              # and the parabolic conoid's parabola is open too
              or (m == 'CONOID'
                  and op.conoid_kind not in ('WHITNEY', 'ZINDLER',
                                             'PARABOLIC_CONOID')))
    return [(rail0, closed), (rail1, closed)]


def _loop_segments(loops):
    """Flatten (points, closed) polylines into consecutive (p0, p1)
    segments so boundary curves can flow through _edges / _rods exactly
    like the rulings do."""
    segs = []
    for pts, closed in loops:
        pts = [tuple(p) for p in pts]
        for i in range(len(pts) - 1):
            segs.append((pts[i], pts[i + 1]))
        if closed and len(pts) > 2:
            segs.append((pts[-1], pts[0]))
    return segs


# --------------------------------------------------------------------
# Surface thickness: crossing check and Fused Solid sizing
# --------------------------------------------------------------------

def count_crossings(tris, pairs):
    """Of the candidate triangle index `pairs` (a BVH overlap test, which
    may list a pair in either order or both), the number of distinct
    pairs sharing no corner -- faces really passing through each other,
    not neighbours meeting along an edge or at a vertex."""
    P = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(P):
        return 0
    P = np.unique(np.sort(P, axis=1), axis=0)
    P = P[P[:, 0] != P[:, 1]]
    T = np.asarray(tris, dtype=np.int64)
    ta, tb = T[P[:, 0]], T[P[:, 1]]
    shared = (ta[:, :, None] == tb[:, None, :]).any(axis=(1, 2))
    return int((~shared).sum())


#: voxels across the wall, and surface samples across it
_FUSED_VOXELS = 8
_FUSED_SAMPLES = 6
#: the finest voxel, as a fraction of the object's size, so a very thin
#: wall on a large surface cannot ask for an unbounded volume
_FUSED_FINEST = 1.0 / 400.0
#: a density ball of radius r meshes about 0.6 voxel inside r at the
#: half-density threshold (measured 0.58-0.62 voxel at 6-10 voxels
#: across the wall), so the radius is padded by that much
_FUSED_PAD = 0.6
#: passes of neighbour averaging that take the voxel grain off
_FUSED_SMOOTHING = 3


def fused_solid_params(thickness, extent):
    """Voxel size, surface sample spacing and ball radius for a Fused
    Solid wall of `thickness` on a surface `extent` across.

    The wall is the set of points within thickness / 2 of the surface.
    It is built by scattering Poisson-disk samples over the surface,
    giving each a ball of density in a sparse volume and meshing the
    volume at half density.  Samples a sixth of the thickness apart
    overlap enough that the union of their balls is smooth to well
    under a voxel; closer than half a voxel adds nothing."""
    voxel = max(thickness / _FUSED_VOXELS, extent * _FUSED_FINEST)
    spacing = max(thickness / _FUSED_SAMPLES, 0.5 * voxel)
    radius = 0.5 * thickness + _FUSED_PAD * voxel
    return voxel, spacing, radius


if _IN_BLENDER:

    _FUSED_GROUP = "Math Art Fused Solid"
    _FUSED_VERSION = 1

    def _self_crossings(context, obj):
        """Pairs of faces of `obj`'s evaluated mesh (its modifiers
        applied) that pass through each other (`count_crossings`)."""
        from mathutils.bvhtree import BVHTree
        dg = context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        m = ev.to_mesh()
        try:
            m.calc_loop_triangles()
            co = np.empty(len(m.vertices) * 3)
            m.vertices.foreach_get('co', co)
            tris = np.empty(len(m.loop_triangles) * 3, dtype=np.int64)
            m.loop_triangles.foreach_get('vertices', tris)
        finally:
            ev.to_mesh_clear()
        tris = tris.reshape(-1, 3)
        if not len(tris):
            return 0
        bvh = BVHTree.FromPolygons(co.reshape(-1, 3).tolist(),
                                   tris.tolist(), all_triangles=True)
        return count_crossings(tris, bvh.overlap(bvh))

    def _set_node_choice(node, prop, prop_value, socket, socket_value):
        """Set a node option that is a node property in older Blenders
        and a menu input in newer ones."""
        if hasattr(node, prop):
            setattr(node, prop, prop_value)
        else:
            node.inputs[socket].default_value = socket_value

    def _fused_solid_group():
        """The shared Fused Solid node group, made on first use, and its
        input identifiers by name.  One group serves every object; each
        modifier carries its own input values."""
        ng = next((g for g in bpy.data.node_groups
                   if g.get("math_art_fused_solid") == _FUSED_VERSION),
                  None)
        if ng is None:
            ng = bpy.data.node_groups.new(_FUSED_GROUP, 'GeometryNodeTree')
            ng["math_art_fused_solid"] = _FUSED_VERSION
            face = ng.interface
            face.new_socket("Geometry", in_out='INPUT',
                            socket_type='NodeSocketGeometry')
            face.new_socket("Geometry", in_out='OUTPUT',
                            socket_type='NodeSocketGeometry')
            for name, kind in (("Point Spacing", 'NodeSocketFloat'),
                               ("Point Density", 'NodeSocketFloat'),
                               ("Radius", 'NodeSocketFloat'),
                               ("Voxel Size", 'NodeSocketFloat'),
                               ("Smoothing", 'NodeSocketInt'),
                               ("Smooth Shading", 'NodeSocketBool')):
                face.new_socket(name, in_out='INPUT', socket_type=kind)
            N, L = ng.nodes, ng.links
            gin = N.new('NodeGroupInput')
            gout = N.new('NodeGroupOutput')
            scatter = N.new('GeometryNodeDistributePointsOnFaces')
            _set_node_choice(scatter, 'distribute_method', 'POISSON',
                             'Distribute Method', 'Poisson Disk')
            splat = N.new('GeometryNodePointsToVolume')
            _set_node_choice(splat, 'resolution_mode', 'VOXEL_SIZE',
                             'Resolution Mode', 'Size')
            splat.inputs['Density'].default_value = 1.0
            mesh = N.new('GeometryNodeVolumeToMesh')
            _set_node_choice(mesh, 'resolution_mode', 'GRID',
                             'Resolution Mode', 'Grid')
            mesh.inputs['Threshold'].default_value = 0.5
            pos = N.new('GeometryNodeInputPosition')
            blur = N.new('GeometryNodeBlurAttribute')
            blur.data_type = 'FLOAT_VECTOR'
            place = N.new('GeometryNodeSetPosition')
            shade = N.new('GeometryNodeSetShadeSmooth')
            # link by socket NAME: interface order is not creation order
            src = {s.name: s for s in gin.outputs}
            vec_in = next(s for s in blur.inputs
                          if s.type == 'VECTOR' and s.enabled)
            vec_out = next(s for s in blur.outputs
                           if s.type == 'VECTOR' and s.enabled)
            L.new(src["Geometry"], scatter.inputs['Mesh'])
            L.new(src["Point Spacing"], scatter.inputs['Distance Min'])
            L.new(src["Point Density"], scatter.inputs['Density Max'])
            L.new(scatter.outputs['Points'], splat.inputs['Points'])
            L.new(src["Radius"], splat.inputs['Radius'])
            L.new(src["Voxel Size"], splat.inputs['Voxel Size'])
            L.new(splat.outputs['Volume'], mesh.inputs['Volume'])
            L.new(mesh.outputs['Mesh'], place.inputs['Geometry'])
            L.new(pos.outputs['Position'], vec_in)
            L.new(src["Smoothing"], blur.inputs['Iterations'])
            L.new(vec_out, place.inputs['Position'])
            L.new(place.outputs['Geometry'], shade.inputs['Geometry'])
            L.new(src["Smooth Shading"], shade.inputs['Shade Smooth'])
            L.new(shade.outputs['Geometry'], gout.inputs[0])
            for k, node in enumerate((gin, scatter, splat, mesh, place,
                                      shade, gout)):
                node.location = (220.0 * k, 0.0)
            pos.location = (440.0, -260.0)
            blur.location = (660.0, -260.0)
        idents = {item.name: item.identifier
                  for item in ng.interface.items_tree
                  if getattr(item, 'in_out', None) == 'INPUT'}
        return ng, idents

    def _add_fused_solid(obj, voxel, spacing, radius, smooth):
        """Give `obj` a live Fused Solid modifier sized by
        `fused_solid_params`.  The inputs are set before the stack first
        evaluates: changing one afterwards needs a depsgraph tag."""
        ng, ids = _fused_solid_group()
        mod = obj.modifiers.new("Fused Solid", 'NODES')
        mod.node_group = ng
        for name, value in (("Point Spacing", spacing),
                            ("Point Density", 4.0 / spacing ** 2),
                            ("Radius", radius), ("Voxel Size", voxel),
                            ("Smoothing", _FUSED_SMOOTHING),
                            ("Smooth Shading", bool(smooth))):
            mod[ids[name]] = value
        return mod

    def _mode_changed(self, context):
        """Carry Output to the new surface's own default -- rods, or the
        filled surface for the compound helical cone -- unless it was set
        by hand (`output_after_mode_change`)."""
        new = output_after_mode_change(self.output, self.mode_seen,
                                       self.mode)
        if new != self.output:
            self.output = new
        if self.mode_seen != self.mode:
            self.mode_seen = self.mode

    class MESH_OT_ruled_surface_add(bpy.types.Operator):
        """Add a ruled surface: a hyperboloid, compound helical
        cone, spiral ruled surface, conoid, tangent developable,
        helicoid, twisted strip or doubly-ruled hyperbolic paraboloid.
        Straight-ruled modes can be rendered as their rulings (rods),
        and those with two crossing ruling families as woven ribbons"""
        bl_idname = "mesh.ruled_surface_add"
        bl_label = "Ruled Surface"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(name="Surface", items=_MODES,
                           default='HYPERBOLOID', update=_mode_changed,
                           description="Which ruled-surface family to "
                                       "build")
        mode_seen: StringProperty(
            name="Output Default For", default='HYPERBOLOID',
            options={'HIDDEN'},
            description="Internal: the surface whose default Output was "
                        "last carried over when the surface changed")
        conoid_kind: EnumProperty(name="Conoid", items=_CONOID_KINDS,
                                  default='PLUCKER',
                                  description="Which right conoid to "
                                              "build (Conoid mode)")

        # shared geometry
        radius: FloatProperty(name="Radius", default=1.0, min=0.01,
                              max=20.0,
                              description="Radius of the base circle or "
                                          "helix the rulings spring from")
        height: FloatProperty(name="Half Height", default=1.0,
                              min=0.05, max=20.0,
                              description="Half the axial height "
                                          "(rings sit at +-this)")
        twist_angle: FloatProperty(
            name="Twist", default=math.radians(120.0), min=0.0,
            max=math.radians(179.0), subtype='ANGLE',
            description="Rotation of the top ring against the bottom "
                        "one.  The waist radius is R cos(twist/2): no "
                        "twist is a cylinder, and it narrows toward a "
                        "double cone as the twist nears 180 degrees")
        family: EnumProperty(
            name="Ruling Family",
            items=[('BOTH', "Both",
                    "Both ruling families together, forming a mesh"),
                   ('RIGHT', "Right",
                    "One family only: the right-handed rulings (twisted "
                    "forward), or one of the saddle's two families"),
                   ('LEFT', "Left",
                    "The other family only: the left-handed rulings "
                    "(twisted back), or the saddle's other family")],
            default='BOTH',
            description="Which ruling family to draw as rods or curves "
                        "(both = the two families together, the mesh of "
                        "a string sculpture)")
        # helical cone
        cone_height: FloatProperty(name="Height", default=2.5,
                                   min=0.1, max=30.0,
                                   description="Total height of the cone, "
                                               "base to apex")
        flutes: IntProperty(name="Flutes", default=6, min=0, max=64,
                            description="Number of helical flutes "
                                        "(ridges) around the cone")
        flute_depth: FloatProperty(name="Flute Depth", default=0.18,
                                   min=0.0, max=0.9,
                                   description="Depth of the flutes at "
                                               "the base, fading to "
                                               "arrises at the apex")
        cone_twist: FloatProperty(name="Flute Turns", default=2.0,
                                  min=-12.0, max=12.0,
                                  description="Turns the flutes wind "
                                              "over the full height")
        taper: FloatProperty(name="Taper", default=1.0, min=0.2,
                             max=4.0,
                             description="Envelope profile exponent "
                                         "(1 = straight cone)")
        orbit_amp: FloatProperty(name="Orbit Amount", default=0.0,
                                 min=0.0, max=1.0,
                                 description="Amplitude of the second "
                                             "(planetary) helix")
        orbit_turns: FloatProperty(name="Orbit Turns", default=1.0,
                                   min=-8.0, max=8.0,
                                   description="Turns of the secondary "
                                               "planetary helix over the "
                                               "height")
        # spiral
        tightness: FloatProperty(name="Spiral Tightness", default=0.15,
                                 min=-0.6, max=0.6,
                                 description="Log-spiral growth k "
                                             "(0 = circle)")
        slope: FloatProperty(name="Ruling Slope", default=1.0,
                             min=-6.0, max=6.0,
                             description="Vertical rise of the rulings; "
                                         "in Helicoid mode tilts them off "
                                         "horizontal (0 = right helicoid)")
        turns: FloatProperty(name="Turns", default=2.0, min=0.1,
                             max=12.0,
                             description="Number of turns of the base "
                                         "curve or helix")
        petals: IntProperty(name="Symmetry", default=1, min=1, max=16,
                            description="Rosette fold count of the "
                                        "base curve (1 = plain "
                                        "spiral)")
        petal_amp: FloatProperty(name="Rosette Amount", default=0.0,
                                 min=0.0, max=1.0,
                                 description="Depth of the rosette lobes "
                                             "on the spiral base curve "
                                             "(0 = plain spiral)")
        # conoid
        amp: FloatProperty(name="Amplitude", default=0.5, min=0.0,
                           max=4.0,
                           description="Height amplitude of the conoid's "
                                       "rise and fall")
        folds: IntProperty(name="Folds", default=3, min=1, max=16,
                           description="Number of lobes for the n-fold "
                                       "and Zindler conoids")
        wallis_a: FloatProperty(
            name="Crest", default=1.0, min=0.05, max=4.0,
            description="The Wallis conical edge's a in "
                        "h = amp sqrt(a^2 - b^2 cos^2 u): its height where "
                        "the edge is tallest.  Gaudi's surface uses it "
                        "for its length, Guimard's for the swing of its "
                        "moving line, and the surface of constant slope "
                        "for its steepness")
        wallis_b: FloatProperty(
            name="Dip", default=0.6, min=0.0, max=4.0,
            description="The Wallis conical edge's b in "
                        "h = amp sqrt(a^2 - b^2 cos^2 u): how far the edge "
                        "dips below its crest.  The surface of constant "
                        "slope uses it for the depth of its base curve's "
                        "lobes")
        # helix-based
        pitch: FloatProperty(name="Pitch", default=0.4, min=-4.0,
                             max=4.0,
                             description="Axial rise per radian")
        v_min: FloatProperty(name="Edge Gap", default=0.04, min=0.0,
                             max=0.5,
                             description="Inner offset from the "
                                         "cuspidal edge")
        inner: FloatProperty(name="Inner Radius", default=0.0,
                             min=0.0, max=10.0,
                             description="Radius of the hole where the "
                                         "helicoid rulings start "
                                         "(0 = full disc)")
        # twist strip
        width: FloatProperty(name="Strip Half-Width", default=0.4,
                             min=0.02, max=5.0,
                             description="Half-width of the twisted "
                                         "strip; keep below the radius")
        half_twists: IntProperty(name="Half Twists", default=1,
                                 min=0, max=12,
                                 description="Number of half-twists; odd "
                                             "gives a one-sided Mobius "
                                             "band")
        # hypar
        hy_a: FloatProperty(name="Width", default=1.0, min=0.05, max=6.0,
                            description="Width scale a in "
                                        "z = c((x/a)^2 - (y/b)^2)")
        hy_b: FloatProperty(name="Depth", default=1.0, min=0.05, max=6.0,
                            description="Width scale b in "
                                        "z = c((x/a)^2 - (y/b)^2)")
        hy_c: FloatProperty(name="Saddle Height", default=1.0,
                            min=0.05, max=6.0,
                            description="Vertical scale c of the saddle")
        use_corners: BoolProperty(name="Corner Points",
                                  default=False,
                                  description="Build the hypar as a "
                                              "bilinear patch spanning "
                                              "four skew corner points")
        p00: FloatVectorProperty(name="Corner 1", size=3,
                                 default=(-1.0, -1.0, -1.0),
                                 description="Corner point of the "
                                             "bilinear patch (s=0, t=0)")
        p10: FloatVectorProperty(name="Corner 2", size=3,
                                 default=(1.0, -1.0, 1.0),
                                 description="Corner point of the "
                                             "bilinear patch (s=1, t=0)")
        p01: FloatVectorProperty(name="Corner 3", size=3,
                                 default=(-1.0, 1.0, 1.0),
                                 description="Corner point of the "
                                             "bilinear patch (s=0, t=1)")
        p11: FloatVectorProperty(name="Corner 4", size=3,
                                 default=(1.0, 1.0, -1.0),
                                 description="Corner point of the "
                                             "bilinear patch (s=1, t=1)")
        # concentric torus-knot span
        knot_p: IntProperty(name="Inner p", default=1, min=1, max=8,
                            description="Times the knots wind around "
                                        "the main axis")
        knot_q: IntProperty(name="Inner q", default=3, min=0, max=9,
                            description="Times the inner knot winds "
                                        "around the tube; 0 makes it a "
                                        "flat circle wound p times")
        knot_scale: FloatProperty(name="Inner Scale", default=1.0,
                                  min=0.1, max=5.0,
                                  description="Overall size of the inner "
                                              "torus knot")
        knot_tube: FloatProperty(name="Inner Tube", default=1.0,
                                 min=0.0, max=5.0,
                                 description="Tube radius of the inner "
                                             "torus knot")
        knot_inner_height: FloatProperty(name="Inner Height",
                                         default=1.0, min=0.0, max=5.0,
                                         description="Vertical "
                                                     "oscillation of "
                                                     "the inner knot")
        knot_inner_lift: FloatProperty(name="Inner Lift", default=0.0,
                                       min=-10.0, max=10.0,
                                       description="Shift the inner "
                                                   "knot up or down "
                                                   "along the axis")
        knot_rotation: FloatProperty(name="Inner Rotation", default=0.0,
                                     min=-_TWO_PI, max=_TWO_PI,
                                     subtype='ANGLE',
                                     description="Rotate the inner knot "
                                                 "about the axis, "
                                                 "twisting the rulings")
        knot_outer_p: IntProperty(name="Outer p", default=0, min=0,
                                  max=8,
                                  description="p of the outer knot; 0 "
                                              "matches the inner p so "
                                              "the rulings line up")
        knot_outer_q: IntProperty(name="Outer q", default=6, min=0,
                                  max=9,
                                  description="q of the outer knot; 0 "
                                              "degenerates it to a "
                                              "circle")
        knot_outer_scale: FloatProperty(name="Outer Scale",
                                        default=2.0, min=0.1, max=10.0,
                                        description="Overall size of the "
                                                    "outer torus knot")
        knot_outer_tube: FloatProperty(name="Outer Tube", default=1.0,
                                       min=0.0, max=5.0,
                                       description="Tube radius of the "
                                                   "outer torus knot")
        knot_outer_height: FloatProperty(name="Outer Height",
                                         default=1.0, min=0.0, max=5.0,
                                         description="Vertical "
                                                     "oscillation of "
                                                     "the outer knot")
        knot_circle_radius: FloatProperty(name="Outer Circle Radius",
                                          default=4.5, min=1.0,
                                          max=20.0,
                                          description="Radius of the "
                                                      "outer circle "
                                                      "when Outer q = 0")
        knot_twist: FloatProperty(
            name="Twist", default=math.radians(30.0), min=-math.pi,
            max=math.pi, subtype='ANGLE',
            description="How far along the outer knot each rod's far end "
                        "slides, in knot parameter (a full turn would "
                        "trace the whole knot).  The right family slides "
                        "forward and the left family back, so with both "
                        "families the rods cross into a mesh, as on a "
                        "stick hyperboloid")
        knot_rods: IntProperty(
            name="Rod Count", default=128, min=3, max=800,
            description="Number of rods strung between the two knots in "
                        "rods or bare-curves output (per family)")

        # shared extents / resolution / output
        v_extent: FloatProperty(name="Ruling Extent", default=1.0,
                                min=0.05, max=8.0,
                                description="Half-length of the "
                                            "rulings / patch extent")
        res_u: IntProperty(
            name="Base Curve Samples", default=120, min=6, max=800,
            description="Samples along the base curve, around the "
                        "surface; on the hyperbolic paraboloid, the samples "
                        "along each side of its square grid")
        res_v: IntProperty(
            name="Ruling Samples", default=20, min=1, max=200,
            description="Samples along each ruling, across the surface "
                        "(the hyperbolic paraboloid's square grid takes "
                        "Base Curve Samples both ways)")
        output: EnumProperty(
            name="Output",
            description="Build the filled surface, its straight "
                        "rulings as solid rods or bare curves, or -- on "
                        "a surface with two crossing ruling families -- "
                        "both families woven together as ribbons.  Every "
                        "surface starts as rods except the compound "
                        "helical cone, which is not straight-ruled and "
                        "starts as its surface; its rods and curves draw "
                        "its arrises, the helical ridges of its flutes",
            items=[('SURFACE', "Surface",
                    "The filled ruled surface"),
                   ('RODS', "Rulings as Rods",
                    "The straight rulings as solid rods (string / "
                    "stick sculpture)"),
                   ('CURVES', "Bare Curves",
                    "The straight rulings as a bare wireframe of "
                    "edges (no faces)"),
                   ('RIBBONS', "Woven Ribbons",
                    "Both ruling families as flat ribbons lying in the "
                    "surface, passing over and under each other where "
                    "they cross.  A surface with only one family of "
                    "rulings weaves them with the curves running along "
                    "it, as a basket's stakes are woven with its "
                    "weavers.  The compound helical cone, which has no "
                    "rulings, falls back to rods")],
            default='RODS')
        n_rods: IntProperty(name="Rod Count", default=48, min=3,
                            max=400,
                            description="Number of rulings drawn in rods, "
                                        "bare-curves or woven-ribbons "
                                        "output (per family)")
        ribbon_width: FloatProperty(
            name="Ribbon Width", default=0.9, min=0.05, max=2.0,
            description="Width of each woven ribbon, as a fraction of "
                        "the widest that still weaves cleanly: 1 leaves "
                        "just enough room between crossings for a ribbon "
                        "to pass from over to under without touching.  On "
                        "the concentric toroidal knots it is measured cell "
                        "by cell, so ribbons widen where the lattice opens "
                        "out.  Above 1 the ribbons are wider than weaves "
                        "cleanly, and may touch where they cross")
        ribbon_thickness: FloatProperty(
            name="Ribbon Thickness", default=0.15, min=0.01, max=1.0,
            description="Thickness of each ribbon as a fraction of its "
                        "width; where two ribbons cross they also part "
                        "by this much")
        weave_float: IntProperty(
            name="Float Length", default=1, min=1, max=4,
            description="Crossings a ribbon passes over before it goes "
                        "under: 1 is a plain weave, 2 a 2/2 twill, 3 a "
                        "3/3 twill")
        ribbon_overhang: FloatProperty(
            name="Overhang", default=0.0, min=-0.45, max=1.0,
            description="Extend every ribbon straight past the edge at "
                        "both ends, by this fraction of the longest "
                        "ruling.  The extended rulings stay on the "
                        "surface, so the weave carries on past the "
                        "boundary curves.  Negative trims the ribbons "
                        "back from the edge instead")
        weave_gap: FloatProperty(
            name="Weave Gap", default=0.2, min=0.02, max=0.45,
            description="How far the woven ribbons stop short of where "
                        "the rulings run together -- a cone's apex, a "
                        "conoid's axis, the edge a tangent developable "
                        "folds along, the seams where Guimard's surface "
                        "and the milk carton pinch shut -- as a fraction "
                        "of the rulings' length.  Many strands meet at "
                        "one point there, and no weave can pass them all "
                        "over and under one another")
        rod_radius: FloatProperty(name="Rod Radius", default=0.02,
                                  min=0.002, max=0.3,
                                  description="Radius of each rod in rods "
                                              "output")
        separate_rods: BoolProperty(
            name="Separate Touching Rods", default=False,
            description="Where two rods would pass through each other, "
                        "bend both aside just far enough to leave a gap "
                        "of a quarter of the rod radius, so the rods make "
                        "a solid model with no intersections.  Rod ends "
                        "stay put, so rods that meet at a boundary curve "
                        "stay joined there")
        show_boundaries: BoolProperty(
            name="Include Boundary Curves", default=True,
            description="Add the directrix / rail curves the rulings "
                        "are strung between (the two torus knots, the "
                        "hyperboloid's end circles, etc.) to the rods / "
                        "bare-curves output")
        smooth: BoolProperty(name="Smooth Shading", default=True,
                             description="Shade the surface smooth")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="Wall thickness given to the surface (Surface "
                        "output); 0 leaves it a bare surface")
        thickness_method: EnumProperty(
            name="Thickness Method",
            items=(('AUTO', "Automatic",
                    "Solidify, unless the solidified wall would fold or "
                    "pass through itself; then Fused Solid"),
                   ('SOLIDIFY', "Solidify",
                    "Offset the surface to both sides along its normals. "
                    "Light and sharp-rimmed, but it folds into fins "
                    "wherever the surface bends more tightly than half "
                    "the thickness, pinches, or passes through itself"),
                   ('FUSED', "Fused Solid",
                    "Keep everything within half the thickness of the "
                    "surface, meshed from a volume: always one closed "
                    "solid, even where the surface pinches or passes "
                    "through itself, with rounded rims and a much "
                    "heavier mesh")),
            default='AUTO',
            description="How the surface is given its thickness")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size; 1 fits the "
                                         "2 m cube")

        def execute(self, context):
            if self.mode_seen != self.mode:
                # built straight from a script or a preset: this is the
                # surface a later change of surface carries Output from
                self.mode_seen = self.mode
            verts, faces, name = _build_surface(self)
            edges = []
            info = ""
            out = effective_output(self)
            want_rulings = out in ('RODS', 'CURVES')
            if self.output == 'RIBBONS' and out != 'RIBBONS':
                info = " [woven ribbons need two crossing ruling families]"
            if out == 'RIBBONS':
                # a weave is BOTH ruling families crossing into a mesh;
                # pin the family to match, so the panel says so and
                # switching back to rods shows that same mesh
                if self.family != 'BOTH':
                    self.family = 'BOTH'
                # an overhang only lengthens the strands; the weave
                # carries on through the extra crossings past the edge
                fa, fb, weave_opts = _weave_input(self)
                verts, faces, plan = weave_rulings(
                    fa, fb, self.ribbon_width, self.ribbon_thickness,
                    self.weave_float, **weave_opts)
                loops = (_boundary_loops(self)
                         if self.show_boundaries else [])
                tv, tf = _tubes(loops, self.rod_radius, 8)
                o = len(verts)
                verts = list(verts) + tv
                faces = list(faces) + [[i + o for i in q] for q in tf]
                name += " (Woven Ribbons)"
                info += f" crossings={len(plan['crossings']['ia'])}"
                squeezed = (plan['tight']
                            if weave_opts.get('local_width') else 0)
                if plan['conflicts'] or plan['tight'] > squeezed:
                    self.report(
                        {'WARNING'},
                        f"{plan['conflicts']} crossings could not "
                        f"alternate, {plan['tight']} spans too tight for "
                        f"the ribbons to part: narrow the ribbons or use "
                        f"fewer rulings")
                elif squeezed:
                    info += (f" [{squeezed} spans squeezed where the "
                             f"strands crowd together]")
            elif want_rulings and self.mode in _RULED:
                segs = _build_rulings(self)
                curves = _build_curves(self)
                loops = (_boundary_loops(self)
                         if self.show_boundaries else [])
                if out == 'RODS':
                    # a curved ruling needs only the points that keep it
                    # true to a tenth of the rod radius
                    curves = [_simplify_polyline(c, 0.1 * self.rod_radius)
                              for c in curves]
                    if self.separate_rods and self.mode != 'HELICAL_CONE':
                        polys, sep = separate_rods(
                            [np.asarray(s, dtype=float) for s in segs]
                            + curves, self.rod_radius)
                        info += f" separated {sep['contacts']} contacts"
                        if sep['remaining']:
                            self.report(
                                {'WARNING'},
                                f"{sep['remaining']} pairs of rods still "
                                f"pass through each other: try a smaller "
                                f"rod radius or fewer rods")
                    else:
                        polys = [np.asarray(s, dtype=float) for s in segs] \
                            + curves
                    # straight rods are capped sticks; bent or curved
                    # ones are swept as tubes
                    verts, faces = _rods([tuple(map(tuple, p)) for p in polys
                                          if len(p) == 2], self.rod_radius, 8)
                    bv, bf = _tubes([(p, False) for p in polys if len(p) > 2],
                                    self.rod_radius, 8)
                    o = len(verts)
                    verts = list(verts) + bv
                    faces = list(faces) + [[i + o for i in q] for q in bf]
                    # the rails are continuous curves, so they are swept
                    # as one tube each instead of a capped cylinder per
                    # chord, which would lump at every joint
                    tv, tf = _tubes(loops, self.rod_radius, 8)
                    o = len(verts)
                    verts = list(verts) + tv
                    faces = list(faces) + [[i + o for i in q] for q in tf]
                    name += " (Rods)"
                else:
                    verts, faces = _edges(
                        segs + _loop_segments(loops)
                        + _loop_segments([(c, False) for c in curves]))
                    name += " (Curves)"
            elif want_rulings:
                info = " [rulings N/A for this mode]"
            if self.mode == 'HYPERBOLOID':
                a = self.radius * math.cos(self.twist_angle / 2.0)
                info += f" waist a={a:.3f}"
            if not verts:
                self.report({'ERROR'}, "empty mesh")
                return {'CANCELLED'}
            # fit within a 2 x scale cube at the origin
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0
                       for k in range(3)) or 1.0
            s = self.scale / half
            verts = [tuple((v[k] - (lo[k] + hi[k]) / 2.0) * s
                           for k in range(3)) for v in verts]
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, edges, faces)
            me.validate(clean_customdata=True)
            if me.polygons:
                me.polygons.foreach_set(
                    'use_smooth', [self.smooth] * len(me.polygons))
            if (self.smooth and out == 'RIBBONS'
                    and hasattr(me, 'set_sharp_from_angle')):
                # a ribbon is a four-sided box: smoothing across its
                # square corners turns the flat straps into pillowy
                # tubes, so keep the corners sharp and only the gentle
                # bends along each ribbon (and the rail tubes) smooth
                me.set_sharp_from_angle(angle=math.radians(60.0))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0 and out == 'SURFACE':
                method = self.thickness_method
                if method != 'FUSED':
                    mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                    mod.thickness = self.thickness
                    mod.offset = 0.0
                    if method == 'AUTO':
                        crossed = _self_crossings(context, obj)
                        if crossed:
                            obj.modifiers.remove(mod)
                            method = 'FUSED'
                            info += (f" [Solidify would pass through "
                                     f"itself ({crossed} face pairs): "
                                     f"Fused Solid used]")
                if method == 'FUSED':
                    voxel, spacing, radius = fused_solid_params(
                        self.thickness, 2.0 * self.scale)
                    _add_fused_solid(obj, voxel, spacing, radius,
                                     self.smooth)
                    info += f" fused solid voxel={voxel:.4f}"
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}{info}")
            return {'FINISHED'}

        def draw(self, context):
            # One layout for every surface: the surface and its shape,
            # then resolution, then the output and only the options that
            # output uses, then the overall scale.
            lay = self.layout
            lay.use_property_split = True
            m = self.mode
            out = effective_output(self)
            lay.prop(self, 'mode')
            for key, text, editable in _shape_controls(self):
                row = lay
                if not editable:
                    row = lay.row()
                    row.enabled = False
                if text is None:
                    row.prop(self, key)
                else:
                    row.prop(self, key, text=text)

            lay.separator()
            if m == 'HYPAR':
                lay.prop(self, 'res_u', text="Grid Samples")
            else:
                lay.prop(self, 'res_u')
                lay.prop(self, 'res_v')

            lay.separator()
            lay.prop(self, 'output')
            if self.output == 'RIBBONS' and out != 'RIBBONS':
                lay.label(text="Woven ribbons need two crossing "
                               "ruling families", icon='INFO')
            count = 'knot_rods' if m == 'KNOT_SPAN' else 'n_rods'
            if out == 'SURFACE':
                lay.prop(self, 'smooth')
                lay.prop(self, 'thickness')
                if self.thickness > 0:
                    lay.prop(self, 'thickness_method')
            elif out in ('RODS', 'CURVES'):
                if m in _TWO_FAMILY:
                    lay.prop(self, 'family')
                lay.prop(self, count)
                if out == 'RODS':
                    lay.prop(self, 'rod_radius')
                    if m != 'HELICAL_CONE':
                        lay.prop(self, 'separate_rods')
                lay.prop(self, 'show_boundaries')
                if out == 'RODS':
                    lay.prop(self, 'smooth')
            else:  # RIBBONS
                if m in _TWO_FAMILY:
                    # always both families: shown, but not editable
                    row = lay.row()
                    row.enabled = False
                    row.prop(self, 'family')
                lay.prop(self, count, text="Ribbon Count")
                for k in ('ribbon_width', 'ribbon_thickness',
                          'weave_float', 'ribbon_overhang'):
                    lay.prop(self, k)
                if _weave_gap_used(self):
                    lay.prop(self, 'weave_gap')
                lay.prop(self, 'show_boundaries')
                if self.show_boundaries:
                    lay.prop(self, 'rod_radius', text="Boundary Radius")
                lay.prop(self, 'smooth')

            lay.separator()
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.ruled_surface_add",
                             icon='MOD_SCREW')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_ruled_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_ruled_surface_add)


def _selftest():
    builds = [
        ("hyperboloid", lambda: build_hyperboloid(res_u=48, res_v=8)),
        ("helical cone", lambda: build_helical_cone(res_u=48,
                                                    res_v=24)),
        ("spiral", lambda: build_spiral_ruled(res_u=64, res_v=8)),
        ("spiral rosette", lambda: build_spiral_ruled(
            tightness=0.0, petals=5, petal_amp=0.4, res_u=64,
            res_v=8)),
        ("plucker", lambda: build_conoid('PLUCKER', res_u=48,
                                         res_v=8)),
        ("nfold", lambda: build_conoid('NFOLD', folds=5, res_u=48,
                                       res_v=8)),
        ("wallis", lambda: build_conoid('WALLIS', res_u=48, res_v=8)),
        ("zindler", lambda: build_conoid('ZINDLER', folds=2, res_u=48,
                                         res_v=8)),
        ("whitney", lambda: build_conoid('WHITNEY', res_u=32,
                                         res_v=8)),
        ("parabolic conoid", lambda: build_conoid('PARABOLIC_CONOID',
                                                  res_u=48, res_v=8)),
        ("sinusoidal cone", lambda: build_conoid('SINUSOIDAL_CONE',
                                                 folds=3, res_u=48,
                                                 res_v=8)),
        ("helicoidal cone", lambda: build_conoid('HELICOIDAL_CONE',
                                                 res_u=64, res_v=8)),
        ("tangent dev", lambda: build_tangent_developable(res_u=64,
                                                          res_v=8)),
        ("helicoid", lambda: build_helicoid(res_u=64, res_v=8)),
        ("oblique helicoid", lambda: build_helicoid(slope=0.6,
                                                    res_u=64,
                                                    res_v=8)),
        ("mobius", lambda: build_twist_strip(half_twists=1, res_u=64,
                                             res_v=4)),
        ("twist annulus", lambda: build_twist_strip(half_twists=2,
                                                    res_u=64,
                                                    res_v=4)),
        ("hypar", lambda: build_hypar(res=24)),
        ("hypar patch", lambda: build_hypar(
            res=24, corners=((-1, -1, -1), (1, -1, 1), (-1, 1, 1),
                             (1, 1, -1)))),
        ("knot span", lambda: build_knot_span(res_u=96, res_v=8)),
        ("knot span circle", lambda: build_knot_span(
            outer_q=0, res_u=96, res_v=8)),
        ("knot span twisted", lambda: build_knot_span(
            inner_rotation=0.7, res_u=96, res_v=8)),
    ]
    for label, fn in builds:
        verts, faces = fn()
        assert len(verts) > 0 and len(faces) > 0, f"{label}: empty"
        finite = all(all(math.isfinite(c) for c in v) for v in verts)
        assert finite, f"{label}: non-finite coordinate"
        valid = all(0 <= i < len(verts) for f in faces for i in f)
        assert valid, f"{label}: face index out of range"
        print(f"{label}: V={len(verts)} F={len(faces)} "
              f"finite={finite} indices_ok={valid}")

    # Zindler's conoid is defined by a cartesian equation, so gate on
    # it rather than on "it meshed": for n = 2 every vertex must
    # satisfy z(x^2 - y^2) = 2 a x y exactly, and for general n the
    # cylindrical z = a tan(n theta).  Sampling the height and
    # inverting the tangent (rather than sampling theta) is what makes
    # the mesh bounded, and this is the check that the inversion is
    # right way round -- a sign slip there produces a plausible
    # crown-shaped surface that is simply a different conoid.
    for n_ in (1, 2, 3):
        a_ = 0.5
        verts, _f = build_conoid('ZINDLER', amp=a_, folds=n_,
                                 extent=1.0, res_u=41, res_v=6)
        V = np.asarray(verts, dtype=float)
        # every ruling passes through the axis, and on Oz -- the
        # conoid's double line -- theta is undefined, so those points
        # carry no cylindrical identity to check
        V = V[np.hypot(V[:, 0], V[:, 1]) > 1e-9]
        th = np.arctan2(V[:, 1], V[:, 0])
        # tan(n theta) blows up at the asymptotes; compare as
        # z cos(n th) == a sin(n th), which is the same identity
        # cleared of its denominator and finite everywhere
        resid = np.abs(V[:, 2] * np.cos(n_ * th) - a_ * np.sin(n_ * th))
        assert np.max(resid) < 1e-9, (n_, float(np.max(resid)))
        if n_ == 1:
            # n = 1 degenerates to the equilateral hyperbolic
            # paraboloid x z = a y
            hyp = np.abs(V[:, 0] * V[:, 2] - a_ * V[:, 1])
            assert np.max(hyp) < 1e-9, float(np.max(hyp))
        if n_ == 2:
            cart = np.abs(V[:, 2] * (V[:, 0] ** 2 - V[:, 1] ** 2)
                          - 2.0 * a_ * V[:, 0] * V[:, 1])
            assert np.max(cart) < 1e-9, float(np.max(cart))
        # 2n branches, each an open patch -- and the height really is
        # bounded, which is the whole point of the sampling
        assert np.max(np.abs(V[:, 2])) <= 2.0 + 1e-9
    print("zindler conoid: z = a tan(n theta) holds for n = 1, 2, 3, "
          "xz = ay at n = 1, z(x^2-y^2) = 2axy at n = 2, height "
          "bounded  OK")

    # the three Ferreol additions are defined by cartesian /
    # cylindrical equations, so gate on those identities too:
    #   parabolic conoid   a^2 z = x (b^2 - y^2)  (b = 1, amp = 1/a^2)
    #   sinusoidal cone    z = k rho cos(n theta)
    #   helicoidal cone    every ruling passes through the apex, and
    #                      the directrix ring is the helix z = k theta
    #                      (checked unwound, ring by ring)
    amp_ = 0.5
    verts, _f = build_conoid('PARABOLIC_CONOID', amp=amp_, extent=1.0,
                             res_u=33, res_v=6)
    V = np.asarray(verts, dtype=float)
    resid = np.abs(V[:, 2] - amp_ * V[:, 0] * (1.0 - V[:, 1] ** 2))
    assert np.max(resid) < 1e-12, float(np.max(resid))
    for n_ in (1, 2, 3):
        verts, _f = build_conoid('SINUSOIDAL_CONE', amp=amp_, folds=n_,
                                 extent=1.0, res_u=48, res_v=6)
        V = np.asarray(verts, dtype=float)
        rho = np.hypot(V[:, 0], V[:, 1])
        keep = rho > 1e-9
        th = np.arctan2(V[keep, 1], V[keep, 0])
        resid = np.abs(V[keep, 2]
                       - amp_ * rho[keep] * np.cos(n_ * th))
        assert np.max(resid) < 1e-9, (n_, float(np.max(resid)))
        assert np.min(np.linalg.norm(V, axis=1)) < 1e-12  # apex is there
    verts, _f = build_conoid('HELICOIDAL_CONE', amp=amp_, extent=1.0,
                             res_u=48, res_v=6, turns=2.0)
    V = np.asarray(verts, dtype=float)
    assert np.min(np.linalg.norm(V, axis=1)) < 1e-12
    rho = np.hypot(V[1:, 0], V[1:, 1])
    # z / rho = amp * v with v the (unwound) helix parameter: recover v
    # from z itself and confirm the angle matches it mod 2 pi
    vpar = V[1:, 2] / (amp_ * np.maximum(rho, 1e-300))
    ang = np.arctan2(V[1:, 1], V[1:, 0])
    wrap = np.abs(((vpar - ang) + math.pi) % _TWO_PI - math.pi)
    assert np.max(wrap) < 1e-6, float(np.max(wrap))
    print("ferreol conoids: a^2 z = x(b^2 - y^2) on the parabolic "
          "conoid, z = k rho cos(n theta) on the sinusoidal cone "
          "(n = 1, 2, 3), apex + unwound helix on the helicoidal "
          "cone  OK")

    # the waist radius of the stick hyperboloid must equal R cos(tw/2)
    R, tw = 1.0, 120.0
    verts, _ = build_hyperboloid(R, 1.0, tw, res_u=180, res_v=2)
    waist = min(math.hypot(v[0], v[1]) for v in verts)
    expect = R * math.cos(math.radians(tw) / 2.0)
    assert abs(waist - expect) < 2e-3, (waist, expect)
    print(f"hyperboloid waist {waist:.4f} == R cos(tw/2) "
          f"{expect:.4f}  OK")

    # developability predicate: tangent developable ~ 0, right
    # helicoid != 0
    u = np.linspace(0.2, _TWO_PI * 2.0, 240)
    R_, p = 1.0, 0.35
    det_td = developable_determinant(
        R_ * np.cos(u), R_ * np.sin(u), p * u,
        -R_ * np.sin(u), R_ * np.cos(u), np.full_like(u, p), u)
    det_he = developable_determinant(
        np.zeros_like(u), np.zeros_like(u), 0.4 * u,
        np.cos(u), np.sin(u), np.zeros_like(u), u)
    # tangent developable: det == 0 analytically (two equal rows);
    # the residual is pure finite-difference error, orders below the
    # helicoid's non-zero det (~= pitch = 0.4)
    assert np.max(np.abs(det_td)) < 5e-3, np.max(np.abs(det_td))
    assert np.median(np.abs(det_he)) > 0.1, np.median(np.abs(det_he))
    assert np.median(np.abs(det_he)) > 50 * np.max(np.abs(det_td))
    print(f"developability: tangent-dev |det|max="
          f"{np.max(np.abs(det_td)):.2e} (~0), "
          f"helicoid |det|med={np.median(np.abs(det_he)):.3f} (>0) OK")

    # rods build a non-empty watertight-ish mesh
    segs = rulings_hyperboloid(n=24)          # BOTH families -> 48
    assert len(segs) == 48
    rv, rf = _rods(segs, 0.02, 8)
    assert len(rv) == len(segs) * 2 * 8
    assert len(rf) == len(segs) * (8 + 2)
    print(f"rods: {len(segs)} segments -> V={len(rv)} F={len(rf)} OK")

    # knot span: one ruling per sample joins inner knot to outer knot
    kseg = rulings_knot_span(n=24)
    assert len(kseg) == 24
    assert all(len(s) == 2 and len(s[0]) == 3 for s in kseg)
    # outer q=0 leaves the outer loop planar (z == 0)
    inner, oc = _knot_span_boundaries(outer_q=0, m=32)
    assert np.max(np.abs(oc[:, 2])) < 1e-9
    print(f"knot span: {len(kseg)} rulings, circle-outer planar OK")

    # twist: sliding every far end k samples along the outer knot must be
    # exactly an index shift of the untwisted rods' far ends -- forward
    # for the straight right rods, back for the left strands -- with the
    # near ends (and so both knots) untouched.  Checked on a knot outer
    # and a circle outer.
    n_, k_ = 24, 3
    sh_ = _TWO_PI * k_ / n_
    for oq in (5, 0):
        base = rulings_knot_span(n=n_, outer_q=oq)
        rt = rulings_knot_span(n=n_, outer_q=oq, family='RIGHT', shift=sh_)
        assert rulings_knot_span(n=n_, outer_q=oq, family='BOTH',
                                 shift=sh_) == rt
        assert rulings_knot_span(n=n_, outer_q=oq, family='LEFT',
                                 shift=sh_) == []
        lt = left_rulings_knot_span(n=n_, outer_q=oq, shift=sh_, samples=32)
        for i in range(n_):
            assert np.allclose(rt[i][0], base[i][0])
            assert np.allclose(lt[i][0], base[i][0], atol=1e-12)
            assert np.allclose(rt[i][1], base[(i + k_) % n_][1], atol=1e-9)
            assert np.allclose(lt[i][-1], base[(i - k_) % n_][1], atol=1e-9)
    # the filled surface is the right family's: its far row is those ends
    rt = rulings_knot_span(n=n_, family='RIGHT', shift=sh_)
    kv, _kf = build_knot_span(res_u=n_, res_v=4, shift=sh_)
    far = np.asarray(kv).reshape(n_, 5, 3)[:, -1]
    assert np.allclose(far, [s[1] for s in rt], atol=1e-9)
    # Two coaxial circles make the surface a hyperboloid of one sheet, and
    # then the left strands must BE its straight left rulings, whatever
    # the winding, twist and inner rotation: every sample on the straight
    # line from the strand's start to its end, which is the right
    # ruling's twist mirrored -- outer(u_j - shift + 2 rho / p).
    tw40 = math.radians(40.0)
    for rho_ in (0.0, 0.5):
        for p_ in (1, 2, 3):
            kw_ = dict(p=p_, q=0, outer_q=0, inner_lift=2.0,
                       inner_rotation=rho_)
            lt = left_rulings_knot_span(n=12, shift=tw40, samples=40, **kw_)
            uj_ = _TWO_PI * np.arange(12) / 12
            a_, _o = _knot_span_at(uj_, uj_, **kw_)
            _i, b_ = _knot_span_at(uj_ - tw40 + 2.0 * rho_ / p_,
                                   uj_ - tw40 + 2.0 * rho_ / p_, **kw_)
            for j in range(12):
                d_ = b_[j] - a_[j]
                tt = np.clip(((lt[j] - a_[j]) @ d_) / (d_ @ d_), 0.0, 1.0)
                off = np.linalg.norm(lt[j] - (a_[j] + tt[:, None] * d_),
                                     axis=1)
                assert off.max() < 1e-9, (p_, rho_, j, off.max())
                assert np.allclose(lt[j][-1], b_[j], atol=1e-9)
    # On knotted rails the left strands bend, but they lie on the surface
    # beside the straight right rulings and cross them on it: right ruling
    # i meets left strand j wherever u_i lies strictly between the
    # strand's start u_j and end u_j - 2 shift (the hyperboloid's rule,
    # p = 2 and no rotation here), and there the rod passes through the
    # strand to within the strand's sampling error.
    try:
        from .weaving.rulings import (_segment_pairs_closest,
                                      _pairwise_closest, segment_crossings)
    except ImportError:
        from weaving.rulings import (_segment_pairs_closest,
                                     _pairwise_closest, segment_crossings)
    n48, tw30 = 48, math.radians(30.0)
    kr = rulings_knot_span(n=n48, family='RIGHT', shift=tw30)
    kl = left_rulings_knot_span(n=n48, shift=tw30, samples=1024)
    lattice = [k for k in range(1, n48)
               if _TWO_PI * k / n48 < 2.0 * tw30 - 1e-9]
    on_surface = 0.0
    for j in range(n48):
        P = kl[j]
        for k in lattice:
            rod = np.asarray(kr[(j - k) % n48], dtype=float)
            m_ = len(P) - 1
            _s, _t, _qa, _qb, dd = _segment_pairs_closest(
                np.repeat(rod[:1], m_, axis=0), np.repeat(rod[1:], m_, axis=0),
                P[:-1], P[1:])
            on_surface = max(on_surface, float(dd.min()))
    assert on_surface < 1e-3, on_surface
    # with both families drawn, the rails still come from ONE family (two
    # clean loops, not a zigzag between interleaved ends), and the rod
    # count is per family
    from types import SimpleNamespace
    kop = SimpleNamespace(
        mode='KNOT_SPAN', family='BOTH', res_u=120, knot_p=2, knot_q=3,
        knot_scale=1.0, knot_tube=1.0, knot_inner_height=1.0,
        knot_inner_lift=0.0, knot_rotation=0.0, knot_outer_p=0,
        knot_outer_q=5, knot_outer_scale=2.0, knot_outer_tube=1.0,
        knot_outer_height=1.0, knot_circle_radius=4.5,
        knot_twist=math.radians(30.0), knot_rods=40, n_rods=48)
    rails_ = _boundary_loops(kop)
    assert [len(p) for p, _c in rails_] == [120, 120]
    ring = np.asarray(rails_[0][0])
    step = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    assert step.max() < 3.0 * np.median(step), "rail zigzags"
    assert len(_build_rulings(kop)) == 40
    assert len(_build_curves(kop)) == 40
    kop.family = 'RIGHT'
    assert _build_curves(kop) == []
    print("knot span twist: an exact slide along the outer knot, right "
          "forward and left back; the surface is the right family's; on "
          "two circles the left strands are the hyperboloid's straight "
          "left rulings (p = 1, 2, 3, with and without rotation); on the "
          "knots they cross every right rod of the lattice on the surface "
          "(within %.1e); rails stay single OK" % on_surface)

    # Separate Touching Rods on both families: straight right rods and
    # curved left strands together (64 per family, 30 degree twist, rod
    # radius 0.02).  Judged by brute force on the result: every piece of
    # every rod against every piece of every other, excusing only the
    # stretch next to an end two rods share -- as long as rods leaving
    # it at their angle stay within a clearance -- so the solver's own
    # search is not marking its own homework.  Ends must not move.
    r_kn = 0.02
    want_kn = 2.25 * r_kn
    k_rods = ([np.asarray(s, dtype=float)
               for s in rulings_knot_span(n=64, family='RIGHT', shift=tw30)]
              + [_simplify_polyline(c, 0.1 * r_kn)
                 for c in left_rulings_knot_span(n=64, shift=tw30)])
    k_polys, k_info = separate_rods(k_rods, r_kn)
    assert k_info['contacts'] > 0 and k_info['remaining'] == 0, k_info
    for P, src_ in zip(k_polys, k_rods):
        assert np.allclose(P[0], src_[0]) and np.allclose(P[-1], src_[-1])
    # every pair of rods whose padded boxes overlap, piece against piece
    reach_kn = 2.0 * r_kn * 1.05
    box_lo = np.array([P.min(axis=0) for P in k_polys]) - reach_kn
    box_hi = np.array([P.max(axis=0) for P in k_polys]) + reach_kn
    boxes = np.all((box_lo[:, None, :] <= box_hi[None, :, :])
                   & (box_lo[None, :, :] <= box_hi[:, None, :]), axis=-1)
    real = []
    for a, b in zip(*np.nonzero(np.triu(boxes, 1))):
        Pa, Pb = k_polys[a], k_polys[b]
        s_, t_, dd = _pairwise_closest(Pa[:-1], np.diff(Pa, axis=0),
                                       Pb[:-1], np.diff(Pb, axis=0))
        close = dd < reach_kn
        if not np.any(close):
            continue
        Ra, Rb = k_rods[a], k_rods[b]
        ends = np.linalg.norm(Ra[[0, -1]][:, None] - Rb[[0, -1]][None, :],
                              axis=-1)
        ea, eb = np.unravel_index(int(np.argmin(ends)), ends.shape)
        if ends[ea, eb] < want_kn:
            ta_ = Ra[1] - Ra[0] if ea == 0 else Ra[-2] - Ra[-1]
            tb_ = Rb[1] - Rb[0] if eb == 0 else Rb[-2] - Rb[-1]
            cos_ = abs(float(ta_ @ tb_)) / float(np.linalg.norm(ta_)
                                                 * np.linalg.norm(tb_))
            sin_ = max(math.sqrt(max(0.0, 1.0 - cos_ * cos_)),
                       math.sin(math.radians(3.0)))
            joint = Ra[0] if ea == 0 else Ra[-1]
            reach_ = want_kn / sin_ + want_kn
            qa = (Pa[:-1][:, None, :]
                  + s_[..., None] * np.diff(Pa, axis=0)[:, None, :])
            qb = (Pb[:-1][None, :, :]
                  + t_[..., None] * np.diff(Pb, axis=0)[None, :, :])
            close &= ~((np.linalg.norm(qa - joint, axis=-1) < reach_)
                       & (np.linalg.norm(qb - joint, axis=-1) < reach_))
        if np.any(close):
            real.append(float(dd[close].min()))
    assert not real, (len(real), min(real) / r_kn)
    n_bent = sum(1 for P, src_ in zip(k_polys, k_rods)
                 if len(P) != len(src_) or not np.allclose(P, src_))
    print("knot span rods: straight and curved separated together -- %d "
          "contacts, %d of %d rods bent by at most %.2f radii, smallest "
          "gap %.2f radii; brute-force check finds no two rods within 2.1 "
          "radii away from their joints, and no end moved OK"
          % (k_info['contacts'], n_bent, len(k_rods),
             k_info['max_offset'] / r_kn, k_info['min_gap'] / r_kn))

    # boundary curves: a closed loop of k points -> k segments, an open
    # one -> k-1; both knots feed through _edges like the rulings do
    segs_closed = _loop_segments([(inner, True), (oc, True)])
    assert len(segs_closed) == 2 * len(inner)
    segs_open = _loop_segments([(inner, False)])
    assert len(segs_open) == len(inner) - 1
    bv, be = _edges(segs_closed)
    assert len(be) == len(segs_closed) and len(bv) == 2 * len(segs_closed)
    print(f"boundary curves: {len(inner)}-pt knots -> "
          f"{len(segs_closed)} closed segments OK")

    # bare-curves output: one edge per ruling, no faces
    ev, ee = _edges(segs)
    assert len(ev) == 2 * len(segs) and len(ee) == len(segs)
    assert all(0 <= a < len(ev) and 0 <= b < len(ev) for a, b in ee)
    # every straight-ruled mode yields rulings for rods / curves
    for md in ('SPIRAL', 'CONOID', 'TANGENT_DEV', 'HELICOID',
               'TWIST_STRIP', 'HYPAR'):
        assert md in _RULED
    print(f"curves: {len(segs)} segments -> V={len(ev)} E={len(ee)} OK")

    # ---- the hypar's rulings actually rule it -------------------------
    # A rod set is only a ruling if the WHOLE rod lies on the surface,
    # so sample each segment's interior, not just its two ends: the
    # bug this replaces joined (-e, y) to (+e, y) at one height, which
    # met the surface exactly at the endpoints and nowhere between.
    # Being doubly ruled is the other half of the claim -- both
    # families must be present, and since the saddle is symmetric under
    # y -> -y they must come out congruent.
    for _a, _b, _c, _e in ((1.0, 1.0, 1.0, 1.0), (1.4, 0.7, 1.0, 1.0),
                           (0.6, 1.3, 2.2, 1.2), (1.0, 1.0, -1.0, 1.7)):
        hsegs = rulings_hypar(_a, _b, _c, _e, None, 24)
        assert hsegs, (_a, _b, _c, _e)
        fam = {'u': [], 'v': []}
        for _p, _q in hsegs:
            _p, _q = np.asarray(_p), np.asarray(_q)
            for _t in np.linspace(0.0, 1.0, 11):
                _x, _y, _z = _p + _t * (_q - _p)
                assert abs(_z - _c * ((_x / _a) ** 2
                                      - (_y / _b) ** 2)) < 1e-12,                     ("hypar ruling off the surface", _a, _b, _c, _e, _t)
                assert abs(_x) <= _e + 1e-9 and abs(_y) <= _e + 1e-9,                     ("hypar ruling outside its domain", _x, _y)
            _du = abs((_p[0] / _a + _p[1] / _b)
                      - (_q[0] / _a + _q[1] / _b))
            fam['u' if _du < 1e-9 else 'v'].append(
                float(np.linalg.norm(_q - _p)))
        assert fam['u'] and fam['v'], ("hypar is singly ruled here",
                                       _a, _b, _c, _e)
        assert (sorted(round(x, 9) for x in fam['u'])
                == sorted(round(x, 9) for x in fam['v'])),             ("hypar ruling families differ", _a, _b, _c, _e)
    print("hypar: both ruling families lie on the surface and are "
          "congruent OK")

    # ---- the five named ruled surfaces --------------------------------
    # Each is checked on the property that DEFINES it, not on its mesh.
    for kd in ('GAUDI', 'GUIMARD', 'MILK_CARTON', 'RULED_CUBIC',
               'CONSTANT_SLOPE'):
        V, F = build_named_ruled(kd, res_u=90, res_v=8)
        assert len(F) > 0 and np.all(np.isfinite(np.asarray(V))), kd
        assert len(rulings_named_ruled(kd, n=24)) == 24, kd

    # Milk carton: the source gives a closed-form volume, which is an
    # independent check on the parametrization rather than on the mesh.
    k, al = 0.5, 1.0
    V, F = build_named_ruled('MILK_CARTON', amp=k, extent=al,
                             res_u=320, res_v=160)
    V = np.asarray(V)
    vol = 0.0
    for f in F:
        for i in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[i]], V[f[i + 1]]
            vol += float(np.dot(a, np.cross(b, c))) / 6.0
    want = 4.0 * math.pi * k * k * al ** 3 / 3.0
    assert abs(abs(vol) - want) < 1e-3 * want, (abs(vol), want)
    print("milk carton: volume %.6f vs the published 4.pi.k^2.a^3/3 = "
          "%.6f OK" % (abs(vol), want))

    # Constant slope: the tangent plane must make the SAME angle with the
    # horizontal everywhere -- that is the definition.  Measured on the
    # analytic tangent plane, not on mesh triangles: a quad only lies in
    # the tangent plane in the limit, so a mesh test would be reporting
    # its own truncation error (it falls off like 1/res, from 1.4e-2 at
    # res 100 to 2.2e-5 at 1600).
    uu = np.linspace(0.0, _TWO_PI, 900, endpoint=False)
    du = 1e-7
    worst = 0.0
    for vv in (0.0, 0.35, 0.7, 1.0):
        A, B = named_ruled_curves('CONSTANT_SLOPE', uu)
        A2, B2 = named_ruled_curves('CONSTANT_SLOPE', uu + du)
        P = A + vv * (B - A)
        Pu = ((A2 + vv * (B2 - A2)) - P) / du
        N = np.cross(Pu, B - A)
        nz = np.abs(N[:, 2] / np.linalg.norm(N, axis=1))
        worst = max(worst, float(nz.max() - nz.min()))
        assert abs(float(nz.mean()) - 1.0 / math.sqrt(2.0)) < 1e-6, nz.mean()
    assert worst < 1e-9, worst
    print("constant slope: tangent plane holds one angle to the "
          "horizontal across the whole surface (spread %.1e) OK" % worst)

    # Ruled cubic: it should satisfy a cubic and NOT a quadric.  Fitting
    # both and looking at the null space says which -- a cubic with a
    # 1-dimensional null space is one surface, not a coincidence.
    uu = np.linspace(-1.4, 1.4, 80)
    A, B = named_ruled_curves('RULED_CUBIC', uu)
    vv = np.linspace(0.0, 1.0, 18)
    P = (A[:, None, :] + vv[None, :, None] * (B - A)[:, None, :])
    P = P.reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]

    def _nullity(deg):
        mon = [(i, j, k) for i in range(deg + 1)
               for j in range(deg + 1 - i) for k in range(deg + 1 - i - j)]
        M = np.stack([P[:, 0] ** i * P[:, 1] ** j * P[:, 2] ** k
                      for i, j, k in mon], 1)
        s = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(s < 1e-6 * s[0])), float(s[-1] / s[0])
    n2, _r2 = _nullity(2)
    n3, r3 = _nullity(3)
    assert n2 == 0, "the ruled cubic collapsed onto a quadric"
    assert n3 == 1, ("expected exactly one cubic", n3)
    print("ruled cubic: no quadric fits, and exactly one cubic does "
          "(residual %.1e) OK" % r3)

    # ---- woven ribbons: the hyperboloid ---------------------------------
    # Right ruling i leaves the bottom circle at angle 2 pi i / n and left
    # ruling j at 2 pi j / n.  Written in polar form, a ruling of twist d
    # from angle a sits at angle a + d/2 + atan(s tan(d/2)) at height
    # z = H s (s = -1 .. 1 between the rails), so the two meet where
    #     2 atan(s tan(tw/2)) = theta,   theta = (2 pi k / n - tw) wrapped
    #                                    into (-pi, pi], k = (j - i) mod n
    # i.e. at s = tan(theta/2) / tan(tw/2), and a crossing exists exactly
    # when that s lies on both segments.  Between the rails that is
    # |theta| <= tw; with an overhang of h longest-rulings every ruling
    # (all equally long here) runs on to |s| = 1 + 2h, past the rails,
    # and the extended rulings still meet on the same quadric.
    # Gate on that count and those heights -- and on every crossing
    # lying on the quadric, radius^2 = R^2 (cos^2(tw/2) + s^2 sin^2(tw/2))
    # -- rather than on "the weave meshed".  Along either strand k rises
    # by one per crossing, so the level IS k up to a constant and the
    # plain weave can never conflict.
    for n_, tw_, oh_ in ((48, 120.0, 0.0), (24, 120.0, 0.0),
                         (36, 75.0, 0.0), (30, 150.0, 0.0),
                         (24, 120.0, 0.3), (36, 75.0, 0.5)):
        twr = math.radians(tw_)
        T_ = math.tan(twr / 2.0)
        fa, fb = extend_families(
            rulings_hyperboloid(1.0, 1.0, tw_, 'RIGHT', n_),
            rulings_hyperboloid(1.0, 1.0, tw_, 'LEFT', n_), oh_)
        wv, wf, plan = weave_rulings(fa, fb)
        X = plan['crossings']
        theta = ((_TWO_PI * np.arange(n_) / n_ - twr + math.pi)
                 % _TWO_PI) - math.pi
        s_all = np.tan(theta / 2.0) / T_
        per = int(np.sum(np.abs(s_all) <= (1.0 + 2.0 * oh_) * (1 + 1e-9)))
        assert len(X['ia']) == n_ * per, (n_, tw_, oh_, len(X['ia']),
                                          n_ * per)
        s_ = s_all[(X['ib'] - X['ia']) % n_]
        assert np.max(np.abs(X['point'][:, 2] - s_)) < 1e-9, (n_, tw_, oh_)
        rho2 = X['point'][:, 0] ** 2 + X['point'][:, 1] ** 2
        want = math.cos(twr / 2.0) ** 2 + (s_ * math.sin(twr / 2.0)) ** 2
        assert np.max(np.abs(rho2 - want)) < 1e-9, (n_, tw_, oh_)
        if oh_ > 0.0:                      # the weave went on past the rails
            assert np.max(np.abs(s_)) > 1.0 + 1e-6, (n_, tw_, oh_)
        assert plan['conflicts'] == 0, (n_, tw_, oh_, plan['conflicts'])
        for st in plan['strands']:
            assert np.all(st['sign'][1:] != st['sign'][:-1]), (n_, tw_)
        assert weave_rulings(fa, fb, run=2)[2]['conflicts'] == 0
        assert np.all(np.isfinite(np.asarray(wv)))
        assert all(0 <= i < len(wv) for f in wf for i in f)
        if (n_, tw_) == (48, 120.0) or oh_ > 0.0:
            cl_ = crossing_clearance(plan)
            assert plan['tight'] == 0 and cl_ > 0.5, (n_, tw_, oh_,
                                                      plan['tight'], cl_)
            if (n_, tw_) == (48, 120.0):   # the operator's defaults
                clear = cl_
    print("woven hyperboloid: crossing count, heights and radii match "
          "the closed form, between the rails and past them with an "
          "overhang; plain weave and 2/2 twill conflict-free; ribbons "
          "clear at the defaults (%.2f thickness) OK" % clear)

    # ---- woven ribbons: the hyperbolic paraboloid -----------------------
    # The Ruling Family split must partition the BOTH rods exactly, the
    # crossings must lie on the saddle, and both saddle forms must weave
    # cleanly at the defaults and with an overhang.  The corner patch
    # spanning these four points is the saddle z = -xy (expand the
    # bilinear form with x = 2s - 1, y = 2t - 1), and both saddles are
    # doubly ruled everywhere, so crossings of the EXTENDED rulings must
    # satisfy the same equation.  How many there are differs, and says
    # something: the patch's rulings run corner edge to corner edge, so
    # every line of one family already meets every line of the other
    # inside it -- (n+1)^2 crossings -- and two lines meet only once, so
    # extending them can add none.  The equation form clips its rulings
    # to a square the lattice is diagonal to, so extended rulings DO meet
    # new partners outside the square.
    corner_pts = ((-1, -1, -1), (1, -1, 1), (-1, 1, 1), (1, 1, -1))
    for corners_, height in ((None, lambda x, y: x * x - y * y),
                             (corner_pts, lambda x, y: -x * y)):
        both = rulings_hypar(1.0, 1.0, 1.0, 1.0, corners_, 48)
        fa = rulings_hypar(1.0, 1.0, 1.0, 1.0, corners_, 48, 'RIGHT')
        fb = rulings_hypar(1.0, 1.0, 1.0, 1.0, corners_, 48, 'LEFT')
        assert len(fa) + len(fb) == len(both) and fa and fb
        assert set(fa) | set(fb) == set(both), "families do not partition"
        counts = []
        for oh_ in (0.0, 0.25):
            _wv, _wf, plan = weave_rulings(*extend_families(fa, fb, oh_))
            X = plan['crossings']
            P = X['point']
            counts.append(len(P))
            assert np.max(np.abs(P[:, 2] - height(P[:, 0], P[:, 1]))) \
                < 1e-9, (corners_ is None, oh_)
            clear = crossing_clearance(plan)
            assert plan['conflicts'] == 0 and plan['tight'] == 0, \
                (corners_ is None, oh_, plan['conflicts'], plan['tight'])
            assert clear > 0.5, (corners_ is None, oh_, clear)
        if corners_ is None:
            assert counts[1] > counts[0], counts
        else:
            assert counts[0] == counts[1] == 49 * 49, counts
        print("woven hypar (%s): %d crossings on the saddle (%d with an "
              "overhang), families partition the rods, conflict-free, "
              "clearance %.2f OK"
              % ("equation" if corners_ is None else "corners",
                 counts[0], counts[1], clear))

    # ---- woven ribbons: the concentric toroidal knots --------------------
    # Its left family is curved, so the crossings come from the closed
    # form in `knot_span_weave` rather than from intersecting segments.
    # On two coaxial circles (p = 1, so each circle is traced once) the
    # strands are straight and plain segment intersection is an
    # independent answer: it must find the same pairs at the same points,
    # arc fractions and normals -- with and without inner rotation and an
    # overhang.
    for rho_, oh_ in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.3), (0.5, 0.3)):
        kw_ = dict(p=1, q=0, outer_q=0, inner_lift=2.0, inner_rotation=rho_)
        kright, kleft, KX, _g = knot_span_weave(n=24, shift=tw40,
                                                overhang=oh_, samples=64,
                                                **kw_)
        KY = segment_crossings(kright,
                               [np.stack([c[0], c[-1]]) for c in kleft])
        assert (sorted(zip(KX['ia'], KX['ib']))
                == sorted(zip(KY['ia'], KY['ib']))), (rho_, oh_)
        ox = np.lexsort((KX['ib'], KX['ia']))
        oy = np.lexsort((KY['ib'], KY['ia']))
        for key in ('point', 'ta', 'tb'):
            assert np.max(np.abs(KX[key][ox] - KY[key][oy])) < 1e-9, \
                (rho_, oh_, key)
        assert np.max(np.abs(np.abs(np.einsum(
            'ij,ij->i', KX['normal'][ox], KY['normal'][oy])) - 1.0)) < 1e-9
    # On the default knots: every strand meets its crossings in lattice
    # order -- the shared turning profile guarantees it -- so a plain
    # weave and a 2/2 twill both alternate without a conflict; every
    # crossing lies on its right rod exactly and on its curved left strand
    # to within the strand's sampling; and with the ribbons sized from a
    # quantile of the spans, only the few spans where the strands crowd
    # together are squeezed.
    n64 = 64
    kright, kleft, KX, kgap = knot_span_weave(n=n64, shift=tw30)
    for i in range(n64):
        sel = np.nonzero(KX['ia'] == i)[0]
        assert np.all(np.diff(KX['ib'][sel[np.argsort(KX['ta'][sel])]])
                      % n64 == 1), ("right", i)
        sel = np.nonzero(KX['ib'] == i)[0]
        assert np.all(np.diff(KX['ia'][sel[np.argsort(KX['tb'][sel])]])
                      % n64 == n64 - 1), ("left", i)
    A_ = np.asarray(kright)
    rod_dir = A_[KX['ia'], 1] - A_[KX['ia'], 0]
    off_right = (np.linalg.norm(np.cross(KX['point'] - A_[KX['ia'], 0],
                                         rod_dir), axis=1)
                 / np.linalg.norm(rod_dir, axis=1))
    assert off_right.max() < 1e-9, off_right.max()
    off_left = 0.0
    for c in range(len(KX['ia'])):
        P = kleft[KX['ib'][c]]
        a_, d_ = P[:-1], np.diff(P, axis=0)
        tt = np.clip(np.einsum('ij,ij->i', KX['point'][c] - a_, d_)
                     / np.einsum('ij,ij->i', d_, d_), 0.0, 1.0)
        off_left = max(off_left, float(np.linalg.norm(
            KX['point'][c] - (a_ + tt[:, None] * d_), axis=1).min()))
    assert off_left < 2e-3, off_left
    kv_, kf_, kplan = weave_rulings(kright, kleft, crossings=KX, gap=kgap,
                                    local_width=True)
    assert kplan['conflicts'] == 0
    assert weave_rulings(kright, kleft, run=2, crossings=KX, gap=kgap,
                         local_width=True
                         )[2]['conflicts'] == 0
    spans = 2 * len(KX['ia']) - 2 * n64
    assert kplan['tight'] < 0.1 * spans, (kplan['tight'], spans)
    # sized cell by cell, the ribbons in the open flares are several
    # times wider than those crowded by the inner knot
    assert (np.percentile(kplan['widths'], 90)
            > 3.0 * np.percentile(kplan['widths'], 10)), kplan['widths']
    assert np.all(np.isfinite(np.asarray(kv_)))
    assert max(max(f) for f in kf_) < len(kv_)
    # and the operator's input for it carries the crossings, gap and
    # quantile through
    kop.family, kop.output = 'BOTH', 'RIBBONS'
    kop.ribbon_overhang, kop.knot_rods = 0.0, 12
    kfa, kfb, kopts = _weave_input(kop)
    assert len(kfa) == len(kfb) == 12
    assert set(kopts) == {'crossings', 'gap', 'local_width'}
    assert effective_output(kop) == 'RIBBONS'
    print("woven knot span: on two circles the closed-form crossings "
          "match plain intersection exactly (with rotation and overhang); "
          "on the knots every strand meets its crossings in lattice order, "
          "plain weave and twill conflict-free, crossings on both strands "
          "(left within %.1e), %d of %d spans squeezed where the strands "
          "crowd OK" % (off_left, kplan['tight'], spans))

    # the output falls back to rods on a singly-ruled mode, and only the
    # doubly-ruled modes answer for their two families
    from types import SimpleNamespace
    for md in ('HYPERBOLOID', 'HYPAR', 'SPIRAL', 'HELICAL_CONE'):
        op_ = SimpleNamespace(
            mode=md, output='RIBBONS', radius=1.0, height=1.0,
            twist_angle=math.radians(120.0), n_rods=12, hy_a=1.0,
            hy_b=1.0, hy_c=1.0, v_extent=1.0, use_corners=False,
            p00=None, p10=None, p01=None, p11=None, tightness=0.15,
            slope=1.0, turns=2.0, petals=1, petal_amp=0.0)
        woven = md in _WOVEN
        assert effective_output(op_) == ('RIBBONS' if woven else 'RODS')
        assert (_ruling_families(op_) is not None) == woven, md
    print("woven output: offered where two ruling families cross, rods "
          "elsewhere OK")

    # the Output each surface starts from, and how it follows a change of
    # surface without discarding one chosen by hand
    assert default_output('HELICAL_CONE') == 'SURFACE'
    assert all(default_output(md) == 'RODS'
               for md, _l, _d in _MODES if md != 'HELICAL_CONE')
    for old, new, before, after in (
            ('HYPERBOLOID', 'HELICAL_CONE', 'RODS', 'SURFACE'),
            ('HELICAL_CONE', 'SPIRAL', 'SURFACE', 'RODS'),
            ('HELICAL_CONE', 'SPIRAL', 'RODS', 'RODS'),
            ('HYPERBOLOID', 'HELICAL_CONE', 'SURFACE', 'SURFACE'),
            ('HYPERBOLOID', 'HELICAL_CONE', 'RIBBONS', 'RIBBONS'),
            ('SPIRAL', 'SPIRAL', 'RODS', 'RODS')):
        got = output_after_mode_change(before, old, new)
        assert got == after, (old, new, before, got)
    # the panel lists every surface's shape controls; the knots list each
    # knot p before q, the twist last, and grey out the outer p while the
    # outer knot is a circle
    for md, _l, _d in _MODES:
        shown = _shape_controls(SimpleNamespace(
            mode=md, conoid_kind='WALLIS', use_corners=False,
            knot_outer_q=5))
        assert shown and all(len(c) == 3 for c in shown), md
    kn = SimpleNamespace(mode='KNOT_SPAN', knot_outer_q=5)
    order = [c[0] for c in _shape_controls(kn)]
    assert (order.index('knot_p') < order.index('knot_q')
            < order.index('knot_outer_p') < order.index('knot_outer_q')
            < order.index('knot_outer_scale')), order
    assert order[-1] == 'knot_twist', order
    kn.knot_outer_q = 0
    shown = {c[0]: c[2] for c in _shape_controls(kn)}
    assert shown['knot_outer_p'] is False and 'knot_circle_radius' in shown
    assert 'knot_outer_scale' not in shown
    print("output defaults: rods, the helical cone a surface, a hand-set "
          "output kept across a change of surface; panel order OK")

    # the spiral ruled surface's left family: on a circle it is the
    # hyperboloid's own straight left rulings, on the hyperboloid
    # x^2 + y^2 - (z / slope)^2 = 1
    circ = left_rulings_spiral(0.0, 1.3, 1.0, 1, 0.0, 0.8, n=24,
                               samples=40)
    bend = 0.0
    for P in circ:
        d = P[-1] - P[0]
        L = float(np.linalg.norm(d))
        if L > 1e-9:
            bend = max(bend, float(np.linalg.norm(
                np.cross(P - P[0], d / L), axis=1).max()))
    allp = np.concatenate(circ)
    quad = allp[:, 0] ** 2 + allp[:, 1] ** 2 - (allp[:, 2] / 1.3) ** 2
    assert bend < 1e-9 and np.abs(quad - 1.0).max() < 1e-9, (bend, quad)
    assert rulings_spiral(n=8, family='LEFT') == []
    assert rulings_spiral(n=8, family='BOTH') == rulings_spiral(n=8)
    # on a spiral, a rosette and both at once the crossings are the same
    # surface point on both strands, and the weave alternates everywhere
    worst_off = 0.0
    for k_, pet_, amp_ in ((0.15, 1, 0.0), (0.0, 5, 0.4), (0.12, 4, 0.3)):
        R_, L_, X_, g_ = spiral_weave(k_, 1.0, 2.0, pet_, amp_, 1.0, n=40,
                                      overhang=0.1)
        assert len(X_['ia']) > 40, len(X_['ia'])
        pa = np.array([R_[i][0] + t * (R_[i][1] - R_[i][0])
                       for i, t in zip(X_['ia'], X_['ta'])])
        assert np.abs(pa - X_['point']).max() < 1e-9
        size = float(np.abs(np.concatenate(L_)).max())
        pb = []
        for j, t in zip(X_['ib'], X_['tb']):
            P = L_[j]
            s = np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(P, axis=0), axis=1))])
            pb.append([np.interp(t * s[-1], s, P[:, c]) for c in range(3)])
        off = float(np.linalg.norm(np.asarray(pb) - X_['point'],
                                   axis=1).max()) / size
        worst_off = max(worst_off, off)
        assert off < 5e-3, (k_, pet_, amp_, off)
        _v, _f, plan_ = weave_rulings(R_, L_, 0.9, 0.15, 1, crossings=X_,
                                      gap=g_, local_width=True)
        assert plan_['conflicts'] == 0, (k_, pet_, amp_,
                                         plan_['conflicts'])
    sop = SimpleNamespace(mode='SPIRAL', tightness=0.15, slope=1.0,
                          turns=2.0, petals=1, petal_amp=0.0, v_extent=1.0,
                          n_rods=12, ribbon_overhang=0.2, family='BOTH',
                          output='RIBBONS')
    sfa, sfb, sopts = _weave_input(sop)
    assert len(sfa) == 12 and len(sfb) >= 12
    assert set(sopts) == {'crossings', 'gap', 'local_width'}
    assert effective_output(sop) == 'RIBBONS'
    assert len(_build_curves(sop)) == len(sfb)
    print("spiral weave: on a circle the left strands are the "
          "hyperboloid's straight rulings; crossings on both strands "
          "(left within %.1e of the size), plain weave conflict-free on a "
          "spiral, a rosette and both OK" % worst_off)

    # the basket weave (`weft_weave`): on a cylinder, a Mobius band with an
    # even and an odd ruling count, and a cone cut short of its apex, the
    # crossings sit exactly on both strands and the plain weave
    # alternates along every strand -- round the seam of every closed
    # weaver too, which the weaver never sees as an edge of its own
    def _flat(fn):
        return _grid_surface(fn)
    cyl = _flat(lambda u, v: np.stack([np.cos(u), np.sin(u), v], -1))
    mob = _flat(lambda u, v: np.stack([
        (1.0 + v * np.cos(u / 2)) * np.cos(u),
        (1.0 + v * np.cos(u / 2)) * np.sin(u), v * np.sin(u / 2)], -1))
    con = _flat(lambda u, v: np.stack([v * np.cos(u), v * np.sin(u),
                                       0.6 * v], -1))
    basket = (
        ("cylinder", _weft_chart(cyl, 11, (0.0, _TWO_PI), 'loop',
                                 (-1.0, 1.0))),
        ("mobius even", _weft_chart(mob, 12, (0.0, _TWO_PI), 'mobius',
                                    (-0.4, 0.4))),
        ("mobius odd", _weft_chart(mob, 11, (0.0, _TWO_PI), 'mobius',
                                   (-0.4, 0.4))),
        ("cone", _weft_chart(con, 16, (0.0, _TWO_PI), 'loop', (0.25, 1.0),
                             (True, False))))
    for label_, ch_ in basket:
        fa_, fb_, X_, g_ = weft_weave([ch_], overhang=0.1)
        if ch_['closure'] == 'loop':
            assert len(fa_) % 2 == 0, label_
        pa = np.array([fa_[i][0] + t * (fa_[i][1] - fa_[i][0])
                       for i, t in zip(X_['ia'], X_['ta'])])
        assert np.abs(pa - X_['point']).max() < 1e-9, label_
        pb = []
        for j, t in zip(X_['ib'], X_['tb']):
            P = fb_[j]
            s = np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(P, axis=0), axis=1))])
            pb.append([np.interp(t * s[-1], s, P[:, c]) for c in range(3)])
        assert np.abs(np.asarray(pb) - X_['point']).max() < 1e-9, label_
        plan_ = plan_weave(fa_, fb_, crossings=X_, gap=g_, local_width=True)
        assert plan_['conflicts'] == 0, (label_, plan_['conflicts'])
        closed = [j for j, P in enumerate(fb_) if np.allclose(P[0], P[-1])]
        assert closed, label_
        for j in closed:
            sel = np.nonzero(X_['ib'] == j)[0]
            sel = sel[np.argsort(X_['tb'][sel])]
            assert (plan_['level'][sel[0]] - plan_['level'][sel[-1]]) % 2, \
                (label_, j)
    # the Whitney umbrella's rods are its rulings, on the surface
    for p0, p1 in rulings_conoid('WHITNEY', extent=1.0, n=8):
        q = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) / 3.0
        w = math.sqrt(max(float(q[2]), 0.0))
        assert min(abs(q[0] - q[1] * w), abs(q[0] + q[1] * w)) < 1e-12
    # and every surface with rulings weaves at its defaults, conflict-free
    base = dict(
        radius=1.0, height=1.0, twist_angle=math.radians(120.0),
        tightness=0.15, slope=1.0, turns=2.0, petals=1, petal_amp=0.0,
        amp=0.5, folds=3, wallis_a=1.0, wallis_b=0.6, pitch=0.4, v_min=0.04,
        inner=0.0, width=0.4, half_twists=1, hy_a=1.0, hy_b=1.0, hy_c=1.0,
        use_corners=False, p00=(-1.0, -1.0, -1.0), p10=(1.0, -1.0, 1.0),
        p01=(-1.0, 1.0, 1.0), p11=(1.0, 1.0, -1.0), knot_p=2, knot_q=3,
        knot_scale=1.0, knot_tube=1.0, knot_inner_height=1.0,
        knot_inner_lift=0.0, knot_rotation=0.0, knot_outer_p=0,
        knot_outer_q=5, knot_outer_scale=2.0, knot_outer_tube=1.0,
        knot_outer_height=1.0, knot_circle_radius=4.5,
        knot_twist=math.radians(30.0), knot_rods=24, v_extent=1.0,
        n_rods=24, family='BOTH', output='RIBBONS', ribbon_overhang=0.1,
        weave_gap=0.2, conoid_kind='PLUCKER')
    combos = ([(md, 'PLUCKER') for md, _l, _d in _MODES
               if md in _WOVEN and md != 'CONOID']
              + [('CONOID', k) for k, _l, _d in _CONOID_KINDS])
    for md, kind in combos:
        op_ = SimpleNamespace(**dict(base, mode=md, conoid_kind=kind))
        fa_, fb_, opts_ = _weave_input(op_)
        plan_ = plan_weave(fa_, fb_, 0.9, 0.15, 1, **opts_)
        assert len(plan_['crossings']['ia']), (md, kind)
        assert plan_['conflicts'] == 0, (md, kind, plan_['conflicts'])
    assert 'HELICAL_CONE' not in _WOVEN
    print("basket weave: crossings exact on both strands, plain weave "
          "alternating round every closed weaver on a cylinder, Mobius "
          "bands of both parities and a cone; Whitney rods on the "
          "surface; all %d surfaces with rulings weave conflict-free OK"
          % len(combos))

    # the crossing count: pairs sharing a corner are neighbours, not
    # crossings, and a pair listed both ways counts once
    ct = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert count_crossings(ct, [(0, 1), (1, 0), (0, 2), (2, 0), (3, 3),
                                (3, 2)]) == 2
    assert count_crossings(ct, []) == 0
    # Fused Solid sizing: an eighth of the wall per voxel, floored by the
    # object's size; samples a sixth apart; the radius padded by the
    # measured threshold shrink
    fv, fs, fr = fused_solid_params(0.06, 2.0)
    assert abs(fv - 0.0075) < 1e-12 and abs(fs - 0.01) < 1e-12, (fv, fs)
    assert abs(fr - (0.03 + _FUSED_PAD * 0.0075)) < 1e-12, fr
    fv, fs, fr = fused_solid_params(0.01, 2.0)
    assert abs(fv - 2.0 * _FUSED_FINEST) < 1e-12, fv
    assert abs(fs - 0.5 * fv) < 1e-12 and fr > 0.005, (fs, fr)
    # ... and why Solidify cannot be trusted on the knot span: an inner
    # (1, 3) knot and an outer (1, 6) knot at twice its scale touch
    # wherever cos 3t = 1/4 (dr = c(4c - 1), dz = s(1 - 4c)), six times
    t_touch = math.acos(0.25) / 3.0
    ts = np.array([sg * t_touch + k * _TWO_PI / 3.0
                   for sg in (1.0, -1.0) for k in range(3)])
    kin, kout = _knot_span_at(ts, ts, p=1, q=3, knot_scale=1.0, tube=1.0,
                              outer_p=0, outer_q=6, outer_scale=2.0,
                              outer_tube=1.0)
    touch = float(np.linalg.norm(kin - kout, axis=1).max())
    assert touch < 1e-12, touch
    kin2, kout2 = _knot_span_at(ts + 0.1, ts + 0.1, p=1, q=3,
                                outer_p=0, outer_q=6, outer_scale=2.0)
    assert np.linalg.norm(kin2 - kout2, axis=1).min() > 1e-3
    print("surface thickness: crossings count distinct corner-free pairs; "
          "Fused Solid voxel, spacing and padded radius sized from the "
          "wall; the (1,3)/(1,6) knot rails touch at six points (within "
          "%.1e) OK" % touch)
    print("RESULT: OK")
