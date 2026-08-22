
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
# Modes
#   HYPERBOLOID   -- hyperboloid of one sheet from straight rulings
#     strung between two coaxial circles, the top circle rotated by a
#     twist angle.  The waist radius a = R cos(twist/2) is set purely
#     by the twist: 0 -> cylinder, 180 deg -> double cone.  Doubly
#     ruled: left- and right-handed families of rulings.
#   HELICAL_CONE  -- the compound helical cone / Solomonic column: a
#     cone that is spirally fluted (N flutes winding helically) and
#     optionally wound a second time along a planetary helix, its
#     flutes fading from ornament at the base to arrises at the apex.
#   SPIRAL        -- Farris' spiral ruled surfaces: the hyperboloid's
#     generating circle replaced by a logarithmic spiral (or an
#     n-fold-symmetric rosette), swept with a tangent+vertical ruling.
#   CONOID        -- right conoids S=(v cos u, v sin u, h(u)):
#     Plucker's conoid / cylindroid (h = c sin 2u), the n-fold
#     generalization, the Wallis conical edge, Zindler's conoid
#     (h = a tan 2u, the cubic ruled surface z(x^2-y^2) = 2axy) and
#     the Whitney umbrella (a pinch-point ruled surface).
#   TANGENT_DEV   -- the tangent developable of a circular helix,
#     T(u,v) = c(u) + v c'(u): a flat-unrollable (developable) surface
#     whose edge of regression is the helix itself.
#   HELICOID      -- the right / oblique helicoid, a radial ruling
#     screwing up the axis; the right helicoid (slope 0) is minimal.
#   TWIST_STRIP   -- an n-half-twist ruled band; odd n is a Mobius
#     band (one-sided), even n an orientable twisted annulus.
#   HYPAR         -- the doubly-ruled hyperbolic paraboloid, either as
#     z = c((x/a)^2 - (y/b)^2) or as the bilinear patch spanning four
#     user-set skew corner points (the surface of any four points in
#     general position).
#   KNOT_SPAN     -- a ruled surface strung between two concentric
#     (p, q) torus knots: straight rulings interpolate S = inner(u)(1-v)
#     + outer(u) v between an inner and an outer toroidal knot sampled on
#     a shared parameter.  The outer curve degenerates to a plain circle
#     (wound p times) when its q = 0.  A straight-ruled cousin of the
#     soap-film "knot to knot" span in the minimal-surface toolkit.
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
# - J. Plucker, "On a New Geometry of Space" (1865); H. Whitney,
#   "The general type of singularity of a set of 2n-1 smooth
#   functions of n variables" (1943).
# - K. Zindler (1866-1934), the conoid z = a tan 2 theta; see
#   R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#   mathcurve.com, chapter "conoide de Zindler", for the cartesian
#   form and the tangentoid-crown directrix used here.  A converted
#   copy of the encyclopedia is in research/books/
#   mathcurve_encyclopedie_formes_mathematiques/.
# - S. A. Coons, "Surfaces for Computer-Aided Design of Space Forms,"
#   MIT Project MAC TR-41 (1967) -- the bilinear patch.
# - Torus knots (p, q): classical; see e.g. C. C. Adams, "The Knot
#   Book" (1994).  The knot-to-knot span is adapted here as a pure
#   ruled surface (cf. the soap-film version in the minimal-surface
#   toolkit).
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
    "description": "Stick hyperboloids, compound helical cones, "
                   "spiral ruled surfaces, conoids, tangent "
                   "developables, helicoids, twisted strips, "
                   "doubly-ruled hypars and concentric torus-knot "
                   "spans -- straight-line-swept surfaces, optionally "
                   "rendered as rulings",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty, FloatVectorProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

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

def _knot_curve(p, q, m, scale=1.0, tube=1.0, major=2.0):
    """A (p, q) torus knot sampled at m points on t in [0, 2pi):
        r     = tube cos(q t) + major
        (x,y) = r (cos(p t), sin(p t)),   z = -tube sin(q t)
    all times `scale`.  q = 0 degenerates to a circle of radius
    (tube + major) wound p times.  endpoint=False so the loop welds
    cleanly under wrap_u."""
    t = np.linspace(0.0, _TWO_PI, m, endpoint=False)
    r = np.cos(q * t) * tube + major
    return np.stack([r * np.cos(p * t), r * np.sin(p * t),
                     -np.sin(q * t) * tube], axis=1) * scale


def _knot_span_boundaries(p=2, q=3, knot_scale=1.0, tube=1.0,
                          inner_height=1.0, inner_lift=0.0,
                          inner_rotation=0.0, outer_p=0, outer_q=5,
                          outer_scale=2.0, outer_tube=1.0,
                          outer_height=1.0, circle_radius=4.5, m=96):
    """Inner and outer boundary loops (each (m, 3)) for the knot span.
    The inner loop is always a (p, q) torus knot, height-scaled, lifted
    and optionally rotated about z.  The outer loop is a second
    (outer_p or p, outer_q) torus knot, or -- when outer_q == 0 -- a
    plain circle of radius `circle_radius` wound p times so the rulings
    still line up."""
    inner = _knot_curve(p, q, m, scale=knot_scale, tube=tube)
    inner[:, 2] *= inner_height
    inner[:, 2] += inner_lift
    if inner_rotation != 0.0:
        ca, sa = math.cos(inner_rotation), math.sin(inner_rotation)
        inner[:, :2] = np.stack(
            [inner[:, 0] * ca - inner[:, 1] * sa,
             inner[:, 0] * sa + inner[:, 1] * ca], axis=1)
    po = outer_p or p
    if outer_q > 0:
        outer = _knot_curve(po, outer_q, m, scale=outer_scale,
                            tube=outer_tube)
        outer[:, 2] *= outer_height
    else:
        t = np.linspace(0.0, _TWO_PI, m, endpoint=False)
        outer = np.stack([circle_radius * np.cos(p * t),
                          circle_radius * np.sin(p * t),
                          np.zeros(m)], axis=1)
    return inner, outer


def build_knot_span(p=2, q=3, knot_scale=1.0, tube=1.0,
                    inner_height=1.0, inner_lift=0.0, inner_rotation=0.0,
                    outer_p=0, outer_q=5, outer_scale=2.0,
                    outer_tube=1.0, outer_height=1.0, circle_radius=4.5,
                    res_u=96, res_v=16):
    """Ruled surface between two concentric torus knots:
        S(u, v) = inner(u) (1 - v) + outer(u) v,   v in [0, 1]
    the same straight-ruling interpolation as the stick hyperboloid,
    with the two coaxial circles replaced by an inner and an outer
    (p, q) torus knot."""
    inner, outer = _knot_span_boundaries(
        p, q, knot_scale, tube, inner_height, inner_lift, inner_rotation,
        outer_p, outer_q, outer_scale, outer_tube, outer_height,
        circle_radius, res_u)
    v = np.linspace(0.0, 1.0, res_v + 1)
    P = np.empty((res_u, res_v + 1, 3))
    for j, vv in enumerate(v):
        P[:, j, :] = inner * (1.0 - vv) + outer * vv
    return _mesh_grid(P, wrap_u=True)


def rulings_knot_span(p=2, q=3, knot_scale=1.0, tube=1.0,
                      inner_height=1.0, inner_lift=0.0,
                      inner_rotation=0.0, outer_p=0, outer_q=5,
                      outer_scale=2.0, outer_tube=1.0, outer_height=1.0,
                      circle_radius=4.5, n=48):
    """Ruling segments of the knot span: one straight rod per sample,
    joining inner(u) to outer(u).  A single ruling family (unlike the
    hyperboloid's crossing left/right pair)."""
    inner, outer = _knot_span_boundaries(
        p, q, knot_scale, tube, inner_height, inner_lift, inner_rotation,
        outer_p, outer_q, outer_scale, outer_tube, outer_height,
        circle_radius, n)
    return [(tuple(inner[i]), tuple(outer[i])) for i in range(n)]


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


def rulings_spiral(tightness=0.15, slope=1.0, turns=2.0, petals=1,
                   petal_amp=0.0, v_extent=1.0, n=64):
    k = tightness
    u = np.linspace(0.0, _TWO_PI * turns, n)
    e = np.exp(k * u)
    p, A = float(petals), petal_amp
    fx = np.cos(u) + A * np.cos(p * u)
    fy = np.sin(u) + A * np.sin(p * u)
    fpx = -np.sin(u) - A * p * np.sin(p * u)
    fpy = np.cos(u) + A * p * np.cos(p * u)
    bx, by = e * fx, e * fy
    dbx = e * (k * fx + fpx)
    dby = e * (k * fy + fpy)
    segs = []
    for i in range(n):
        p0 = (bx[i] - v_extent * dbx[i], by[i] - v_extent * dby[i],
              -slope * v_extent)
        p1 = (bx[i] + v_extent * dbx[i], by[i] + v_extent * dby[i],
              slope * v_extent)
        segs.append((p0, p1))
    return segs


# --------------------------------------------------------------------
# 4. conoids  (right conoid S = (v cos u, v sin u, h(u)))
# --------------------------------------------------------------------

def build_conoid(kind='PLUCKER', amp=0.5, folds=2, wallis_a=1.0,
                 wallis_b=0.6, extent=1.0, res_u=120, res_v=16):
    """Right conoids and the Whitney umbrella.
        PLUCKER  h = amp sin(2u)            (Plucker's cylindroid)
        NFOLD    h = amp sin(folds*u)       (n-leaved conoid)
        WALLIS   h = amp sqrt(a^2 - b^2 cos^2 u)   (Wallis conical edge)
        ZINDLER  h = amp tan(folds*u)       (Zindler's conoid)
        WHITNEY  S = (u v, u, v^2)          (pinch-point umbrella)"""
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
                   wallis_b=0.6, extent=1.0, n=48):
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
        segs = []
        for i in range(n + 1):
            s = -extent + 2.0 * extent * i / n
            segs.append(((s * -extent, s, extent ** 2),
                         (s * extent, s, extent ** 2)))
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
                  n=16):
    segs = []
    if corners is not None:
        P00, P10, P01, P11 = (np.asarray(p, dtype=float)
                              for p in corners)
        for i in range(n + 1):
            s = i / n
            e0 = (1 - s) * P00 + s * P10
            e1 = (1 - s) * P01 + s * P11
            segs.append((tuple(e0), tuple(e1)))        # t-rulings
            f0 = (1 - s) * P00 + s * P01
            f1 = (1 - s) * P10 + s * P11
            segs.append((tuple(f0), tuple(f1)))        # s-rulings
        return segs
    # rulings of z=c((x/a)^2-(y/b)^2) run along x/a +- y/b = const;
    # sample straight lines of each family
    for i in range(n + 1):
        f = -extent + 2.0 * extent * i / n
        # family 1: y fixed sweep x  ->  not straight; use diagonal
        # lines s -> (s, f, ...) are straight only in xy-square edges;
        # emit the two boundary-anchored straight generators instead
        segs.append(((-extent, f, c * ((extent / a) ** 2
                                       - (f / b) ** 2)),
                     (extent, f, c * ((extent / a) ** 2
                                      - (f / b) ** 2))))
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
    ('HYPERBOLOID', "Stick Hyperboloid",
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
]

# modes that are genuinely straight-ruled -> rods available
_RULED = {'HYPERBOLOID', 'SPIRAL', 'CONOID', 'TANGENT_DEV',
          'HELICOID', 'TWIST_STRIP', 'HYPAR', 'KNOT_SPAN'}


def _build_surface(op):
    """(verts, faces, name) for the operator's current mode."""
    m = op.mode
    if m == 'HYPERBOLOID':
        vf = build_hyperboloid(op.radius, op.height, op.twist,
                               op.res_u, op.res_v)
        return (*vf, "Stick Hyperboloid")
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
                          op.res_u, op.res_v)
        return (*vf, "Conoid")
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
                             op.res_u, op.res_v)
        return (*vf, "Concentric Toroidal Knots")
    # HYPAR
    corners = None
    if op.use_corners:
        corners = (op.p00, op.p10, op.p01, op.p11)
    vf = build_hypar(op.hy_a, op.hy_b, op.hy_c, op.v_extent,
                     op.res_v * 3, corners)
    return (*vf, "Hyperbolic Paraboloid")


def _build_rulings(op, n=None):
    """Ruling segments for the current mode (rods mode)."""
    m = op.mode
    if n is None:
        n = op.n_rods
    if m == 'HYPERBOLOID':
        return rulings_hyperboloid(op.radius, op.height, op.twist,
                                   op.family, n)
    if m == 'SPIRAL':
        return rulings_spiral(op.tightness, op.slope, op.turns,
                              op.petals, op.petal_amp, op.v_extent, n)
    if m == 'CONOID':
        return rulings_conoid(op.conoid_kind, op.amp, op.folds,
                              op.wallis_a, op.wallis_b, op.v_extent, n)
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
                                 op.knot_circle_radius, n)
    if m == 'HYPAR':
        corners = (op.p00, op.p10, op.p01, op.p11) \
            if op.use_corners else None
        return rulings_hypar(op.hy_a, op.hy_b, op.hy_c, op.v_extent,
                             corners, n)
    return []


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
    N = max(8, op.res_u)
    if m == 'HYPAR':
        return _hypar_boundary(op, N)
    if m == 'HYPERBOLOID':
        # a single ruling family gives clean bottom/top rails
        segs = rulings_hyperboloid(op.radius, op.height, op.twist,
                                   'RIGHT', N)
    else:
        segs = _build_rulings(op, N)
    if not segs:
        return []
    rail0 = [s[0] for s in segs]
    rail1 = [s[1] for s in segs]
    closed = (m in ('HYPERBOLOID', 'KNOT_SPAN', 'TWIST_STRIP')
              # Zindler's directrix has 2n separate branches running off
              # to infinity, so its rails are open like Whitney's
              or (m == 'CONOID'
                  and op.conoid_kind not in ('WHITNEY', 'ZINDLER')))
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


if _IN_BLENDER:

    class MESH_OT_ruled_surface_add(bpy.types.Operator):
        """Add a ruled surface: a stick hyperboloid, compound helical
        cone, spiral ruled surface, conoid, tangent developable,
        helicoid, twisted strip or doubly-ruled hyperbolic paraboloid.
        Straight-ruled modes can be rendered as their rulings (rods)"""
        bl_idname = "mesh.ruled_surface_add"
        bl_label = "Ruled Surface"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(name="Surface", items=_MODES,
                           default='HYPERBOLOID',
                           description="Which ruled-surface family to "
                                       "build")
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
        twist: FloatProperty(name="Twist", default=120.0, min=0.0,
                             max=179.0,
                             description="Rotation of the top ring "
                                         "in degrees; waist radius "
                                         "= R cos(twist/2)")
        family: EnumProperty(name="Ruling Family",
                             items=[('BOTH', "Both", ""),
                                    ('RIGHT', "Right", ""),
                                    ('LEFT', "Left", "")],
                             default='BOTH',
                             description="Which ruling family to draw "
                                         "as rods (both = crossing "
                                         "string sculpture)")
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
        cone_twist: FloatProperty(name="Spiral Twist", default=2.0,
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
        wallis_a: FloatProperty(name="Wallis a", default=1.0,
                                min=0.05, max=4.0,
                                description="Wallis conical edge "
                                            "parameter a, in "
                                            "sqrt(a^2 - b^2 cos^2 u)")
        wallis_b: FloatProperty(name="Wallis b", default=0.6,
                                min=0.0, max=4.0,
                                description="Wallis conical edge "
                                            "parameter b, in "
                                            "sqrt(a^2 - b^2 cos^2 u)")
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
        hy_a: FloatProperty(name="a", default=1.0, min=0.05, max=6.0,
                            description="Width scale a in "
                                        "z = c((x/a)^2 - (y/b)^2)")
        hy_b: FloatProperty(name="b", default=1.0, min=0.05, max=6.0,
                            description="Width scale b in "
                                        "z = c((x/a)^2 - (y/b)^2)")
        hy_c: FloatProperty(name="Saddle Height", default=1.0,
                            min=0.05, max=6.0,
                            description="Vertical scale c of the saddle")
        use_corners: BoolProperty(name="From 4 Corner Points",
                                  default=False,
                                  description="Build the hypar as a "
                                              "bilinear patch spanning "
                                              "four skew corner points")
        p00: FloatVectorProperty(name="P00", size=3,
                                 default=(-1.0, -1.0, -1.0),
                                 description="Corner point of the "
                                             "bilinear patch (s=0, t=0)")
        p10: FloatVectorProperty(name="P10", size=3,
                                 default=(1.0, -1.0, 1.0),
                                 description="Corner point of the "
                                             "bilinear patch (s=1, t=0)")
        p01: FloatVectorProperty(name="P01", size=3,
                                 default=(-1.0, 1.0, 1.0),
                                 description="Corner point of the "
                                             "bilinear patch (s=0, t=1)")
        p11: FloatVectorProperty(name="P11", size=3,
                                 default=(1.0, 1.0, -1.0),
                                 description="Corner point of the "
                                             "bilinear patch (s=1, t=1)")
        # concentric torus-knot span
        knot_p: IntProperty(name="Knot p", default=2, min=1, max=8,
                            description="Times the knots wind around "
                                        "the main axis")
        knot_q: IntProperty(name="Knot q", default=3, min=0, max=9,
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
        knot_outer_q: IntProperty(name="Outer q", default=5, min=0,
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

        # shared extents / resolution / output
        v_extent: FloatProperty(name="Ruling Extent", default=1.0,
                                min=0.05, max=8.0,
                                description="Half-length of the "
                                            "rulings / patch extent")
        res_u: IntProperty(name="Resolution U", default=120, min=6,
                           max=800,
                           description="Samples around the surface, along "
                                       "the base curve")
        res_v: IntProperty(name="Resolution V", default=20, min=1,
                           max=200,
                           description="Samples across the rulings")
        output: EnumProperty(
            name="Output",
            description="Build the filled surface, or its straight "
                        "rulings as solid rods or bare curves",
            items=[('SURFACE', "Surface",
                    "The filled ruled surface"),
                   ('RODS', "Rulings as Rods",
                    "The straight rulings as solid rods (string / "
                    "stick sculpture)"),
                   ('CURVES', "Bare Curves",
                    "The straight rulings as a bare wireframe of "
                    "edges (no faces)")],
            default='SURFACE')
        n_rods: IntProperty(name="Rod Count", default=48, min=3,
                            max=400,
                            description="Number of rulings drawn in rods "
                                        "or bare-curves output")
        rod_radius: FloatProperty(name="Rod Radius", default=0.02,
                                  min=0.002, max=0.3,
                                  description="Radius of each rod in rods "
                                              "output")
        show_boundaries: BoolProperty(
            name="Include Boundary Curves", default=True,
            description="Add the directrix / rail curves the rulings "
                        "are strung between (the two torus knots, the "
                        "hyperboloid's end circles, etc.) to the rods / "
                        "bare-curves output")
        smooth: BoolProperty(name="Smooth Shading", default=True,
                             description="Shade the surface smooth")
        thickness: FloatProperty(name="Thickness", default=0.0,
                                 min=0.0, max=1.0,
                                 description="Solidify thickness "
                                             "(surface modes)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size; 1 fits the "
                                         "2 m cube")

        def execute(self, context):
            verts, faces, name = _build_surface(self)
            edges = []
            info = ""
            want_rulings = self.output in ('RODS', 'CURVES')
            if want_rulings and self.mode in _RULED:
                segs = _build_rulings(self)
                if self.show_boundaries:
                    segs = segs + _loop_segments(_boundary_loops(self))
                if self.output == 'RODS':
                    verts, faces = _rods(segs, self.rod_radius, 8)
                    name += " (Rods)"
                else:
                    verts, faces = _edges(segs)
                    name += " (Curves)"
            elif want_rulings:
                info = " [rulings N/A for this mode]"
            if self.mode == 'HYPERBOLOID':
                a = self.radius * math.cos(
                    math.radians(self.twist) / 2.0)
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
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0 and self.output == 'SURFACE':
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}{info}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            m = self.mode
            if m == 'HYPERBOLOID':
                keys = ('radius', 'height', 'twist')
            elif m == 'HELICAL_CONE':
                keys = ('radius', 'cone_height', 'flutes',
                        'flute_depth', 'cone_twist', 'taper',
                        'orbit_amp', 'orbit_turns')
            elif m == 'SPIRAL':
                keys = ('tightness', 'slope', 'turns', 'petals',
                        'petal_amp', 'v_extent')
            elif m == 'CONOID':
                lay.prop(self, 'conoid_kind')
                keys = ('amp',)
                if self.conoid_kind in ('NFOLD', 'ZINDLER'):
                    keys += ('folds',)
                elif self.conoid_kind == 'WALLIS':
                    keys += ('wallis_a', 'wallis_b')
                keys += ('v_extent',)
            elif m == 'TANGENT_DEV':
                keys = ('radius', 'pitch', 'turns', 'v_extent',
                        'v_min')
            elif m == 'HELICOID':
                keys = ('radius', 'inner', 'pitch', 'turns', 'slope')
            elif m == 'TWIST_STRIP':
                keys = ('radius', 'width', 'half_twists')
            elif m == 'KNOT_SPAN':
                keys = ('knot_p', 'knot_q', 'knot_scale', 'knot_tube',
                        'knot_inner_height', 'knot_inner_lift',
                        'knot_rotation', 'knot_outer_q')
                if self.knot_outer_q > 0:
                    keys += ('knot_outer_p', 'knot_outer_scale',
                             'knot_outer_tube', 'knot_outer_height')
                else:
                    keys += ('knot_circle_radius',)
            else:  # HYPAR
                lay.prop(self, 'use_corners')
                if self.use_corners:
                    keys = ('p00', 'p10', 'p01', 'p11')
                else:
                    keys = ('hy_a', 'hy_b', 'hy_c', 'v_extent')
            for k in keys:
                lay.prop(self, k)
            lay.separator()
            lay.prop(self, 'res_u')
            lay.prop(self, 'res_v')
            if m in _RULED:
                lay.prop(self, 'output')
                if self.output in ('RODS', 'CURVES'):
                    if m == 'HYPERBOLOID':
                        lay.prop(self, 'family')
                    lay.prop(self, 'n_rods')
                    if self.output == 'RODS':
                        lay.prop(self, 'rod_radius')
                    lay.prop(self, 'show_boundaries')
            for k in ('smooth', 'thickness', 'scale'):
                lay.prop(self, k)

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
    print("RESULT: OK")
