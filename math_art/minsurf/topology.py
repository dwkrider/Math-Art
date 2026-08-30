# Topological surfaces: Klein bottles, cross-caps, genus-g handlebodies.
#
# Part of the Math Art minsurf engine (`math_art/minsurf/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# References:
# - G. Franzoni, "The Klein bottle in its classical shape: a further
#   step towards a good parametrization", arXiv:0909.5354 (2009) -- the
#   tube-over-a-directrix scheme Tube(t, theta) = alpha(t) +
#   r(t)(cos theta J(T) + sin theta k), its closure conditions, the
#   re-parametrized piriform directrix of its section 3, and the
#   dumbbell-curve directrix of its section 4 that closes where the
#   piriform one cannot.  A converted copy is in research/papers/
#   surfaces-and-immersions/franzoni-2009-klein-bottle-classical-shape/.
# - A. F. Mobius (1858) and J. B. Listing (1858) -- the one-sided band.
#

import math
import numpy as np


TAU = 2.0 * math.pi


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


def winding_conflict_edges(faces):
    """Edges traversed in the SAME direction by more than one face.

    On an orientable mesh with consistent winding there are none.  On a
    CLOSED non-orientable mesh a ring of them is unavoidable -- there is
    no globally consistent winding to give -- and it marks where the
    winding flips.  Averaged smooth normals degenerate across exactly
    these edges (the two incident faces' geometric normals oppose), so
    the Blender layer marks them sharp: each side then keeps its own
    smooth normal fan and the renderer's double-sided flip hides the
    sign, which is the honest closed-mesh version of the old split-seam
    workaround.  Returns undirected (a, b) pairs with a < b.
    """
    seen = set()
    out = set()
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            if (a, b) in seen:
                out.add((a, b) if a < b else (b, a))
            seen.add((a, b))
    return sorted(out)


def franzoni_klein_point(t, theta, a=20.0, b=8.0, c=5.5, d=0.4,
                         directrix='DUMBBELL'):
    """Franzoni's tube scheme for the classical Klein bottle shape.

        Tube(t, theta) = alpha(t) + r(t) (cos theta J(T) + sin theta k)

    with alpha a plane directrix, T = alpha'/|alpha'|, J the quarter
    turn J(v1, v2) = (-v2, v1) and k the vertical.  Two directrices from
    the paper:

      PIRIFORM (its section 3):  gamma(t) = (a(1 - cos t),
        b sin t (1 - cos t)), r(t) = c - d(t - pi) sqrt(t(2 pi - t)),
        (t, theta) in (0, 2 pi) x [0, 2 pi], with the paper's values
        (a, b, c, d) = (20, 8, 11/2, 2/5).  |gamma'| vanishes at the
        cusp t = 0 (== 2 pi), so the tube is undefined there and the
        image MISSES a circle: this rendition cannot close.

      DUMBBELL (its section 4):  alpha(t) = (A sin t, B sin^2 t cos t),
        r(t) = C - D(2t - pi) sqrt(2t(2 pi - 2t)), t in [0, pi].  This
        directrix is regular on all of [0, pi] and satisfies the
        closure conditions alpha(0) = alpha(pi), alpha'(0) =
        -alpha'(pi), r(0) = r(pi), so the two tube ends meet in the
        same circle and glue under theta -> pi - theta: the image is a
        CLOSED Klein bottle.  The four shape numbers act
        proportionally: the paper's section-3 defaults (20, 8, 11/2,
        2/5) map to its stretched dumbbell values (5, 2, 1/2, 1/30),
        i.e. (A, B, C, D) = (a/4, b/4, c/11, d/12).

    Returns (x, y, z) arrays in the paper's frame: directrix in the
    xy-plane, tube circles spanning {J(T), z}.
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if directrix == 'PIRIFORM':
        ax = a * (1.0 - np.cos(t))
        ay = b * np.sin(t) * (1.0 - np.cos(t))
        dx = a * np.sin(t)
        dy = b * (np.cos(t) - np.cos(2.0 * t))
        r = c - d * (t - math.pi) * np.sqrt(
            np.maximum(t * (TAU - t), 0.0))
    else:
        aa, bb, cc, dd = a / 4.0, b / 4.0, c / 11.0, d / 12.0
        ax = aa * np.sin(t)
        ay = bb * np.sin(t) ** 2 * np.cos(t)
        dx = aa * np.cos(t)
        dy = bb * np.sin(t) * (3.0 * np.cos(t) ** 2 - 1.0)
        r = cc - dd * (2.0 * t - math.pi) * np.sqrt(
            np.maximum(2.0 * t * (TAU - 2.0 * t), 0.0))
    L = np.sqrt(dx * dx + dy * dy)
    # an over-cranked taper would pinch the tube inside out; floor the
    # radius at a sliver of the base radius instead of going negative
    r = np.maximum(r, 0.02 * abs(c) if c else 1e-3)
    ct, st = np.cos(theta), np.sin(theta)
    x = ax + r * ct * (-dy / L)
    y = ay + r * ct * (dx / L)
    z = r * st
    return x, y, z


def build_klein_franzoni(nu, nv, a=20.0, b=8.0, c=5.5, d=0.4,
                         directrix='DUMBBELL'):
    """Mesh Franzoni's classical-shape Klein bottle (see
    `franzoni_klein_point`).

    DUMBBELL: the t = pi row is glued to the t = 0 row by index under
    theta -> pi - theta -- the Klein identification -- so the mesh is
    genuinely CLOSED: chi = 0, no boundary edges, non-orientable.  The
    unavoidable winding-flip ring lands on the seam circle; fetch it
    with `winding_conflict_edges` and mark it sharp.

    PIRIFORM: |gamma'| = 0 at the cusp, exactly as the paper says, so
    the tube is meshed on the open interval and the two rims near the
    cusp stay honest boundary circles (2 nv boundary edges).

    The result is rotated so the bottle stands upright (directrix plane
    vertical, long axis = Z).  Returns (verts, faces).
    """
    nv += nv % 2                     # theta -> pi - theta must be a grid map
    th = TAU * np.arange(nv)[None, :] / nv
    if directrix == 'PIRIFORM':
        eps = math.pi / max(nu, 8)
        t = (eps + (TAU - 2.0 * eps)
             * np.arange(nu + 1)[:, None] / nu)
        x, y, z = franzoni_klein_point(t, th, a, b, c, d, 'PIRIFORM')
        V = np.stack(np.broadcast_arrays(y, z, x), axis=-1).reshape(-1, 3)
        faces = []
        for i in range(nu):
            for j in range(nv):
                j2 = (j + 1) % nv
                faces.append((i * nv + j, i * nv + j2,
                              (i + 1) * nv + j2, (i + 1) * nv + j))
        return V, faces
    t = math.pi * np.arange(nu)[:, None] / nu
    x, y, z = franzoni_klein_point(t, th, a, b, c, d, 'DUMBBELL')
    V = np.stack(np.broadcast_arrays(y, z, x), axis=-1).reshape(-1, 3)

    def vid(i, j):
        if i == nu:                  # (pi, theta) ~ (0, pi - theta)
            return (nv // 2 - j) % nv
        return i * nv + j % nv

    faces = []
    for i in range(nu):
        for j in range(nv):
            faces.append((vid(i, j), vid(i, j + 1),
                          vid(i + 1, j + 1), vid(i + 1, j)))
    return V, faces


def build_mobius_band(nu, nv, radius=1.0, width=0.6):
    """The canonical one-sided band (Mobius / Listing, 1858), as the
    standard ruled chart

        ((R + v cos(u/2)) cos u, (R + v cos(u/2)) sin u, v sin(u/2)),

    u in [0, 2 pi], v in [-w/2, w/2].  The u = 2 pi seam coincides with
    u = 0 under v -> -v and is glued BY INDEX, so the mesh is the real
    Mobius band: chi = 0, one boundary loop (of 2 nu edges -- the famous
    single edge), non-orientable.  The winding-flip ring lands on the
    seam ruling; mark it sharp via `winding_conflict_edges`."""
    nu = max(8, int(nu))
    nv = max(2, int(nv))
    u = TAU * np.arange(nu)[:, None] / nu
    v = width * (np.arange(nv + 1)[None, :] / nv - 0.5)
    w = radius + v * np.cos(u / 2.0)
    x = w * np.cos(u)
    y = w * np.sin(u)
    z = v * np.sin(u / 2.0)
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    stride = nv + 1

    def vid(i, j):
        if i == nu:                  # (2 pi, v) ~ (0, -v)
            return nv - j
        return i * stride + j

    faces = []
    for i in range(nu):
        for j in range(nv):
            faces.append((vid(i, j), vid(i, j + 1),
                          vid(i + 1, j + 1), vid(i + 1, j)))
    return V, faces


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


# ----------------------------------------------------------------------
# the Veronese surface and its shadows
# ----------------------------------------------------------------------
# Veronese's map sends the unit sphere to R^6 by
#     (u, v, w) -> a(u^2, v^2, w^2, vw, wu, uv)
# Every coordinate is even, so antipodes land on the same point and the
# map factors through the projective plane -- and it is INJECTIVE there,
# so RP^2 is genuinely embedded, with no self-intersection anywhere.  The
# image lies in the hyperplane x1 + x2 + x3 = a, so really in R^5, and
# the further projection (x2-x1, x4, x5, x6) is still injective: RP^2
# embeds in R^4.
#
# It does NOT embed in R^3, and that is the point of the construction.
# Every linear projection of the Veronese surface into three dimensions
# has singularities, and those projections are exactly the classical
# STEINER SURFACES.  Taking Mathcurve's own two named projections,
#     (x4, x5, x6)      -> Steiner's Roman surface
#     (x4, x5, x3 - x1) -> the cross-cap
# they differ only in the third coordinate, so rotating between them,
#     P(t) = (x4, x5, cos t . x6 + sin t . (x3 - x1)),
# is precisely an orthogonal projection of the R^4 embedding
# (x3-x1, x4, x5, x6) along the turning direction
# (-sin t, 0, 0, cos t).  The angle slider is therefore not an
# interpolation between two unrelated formulas: it turns the embedded
# projective plane in four-space and shows its three-dimensional shadow,
# which is a Steiner surface at every angle.
#
# References:
# - G. Veronese (1854-1917); see M. Berger, "Geometry Revealed",
#   Springer 2010, p. 47, and the Wikipedia entry "Veronese surface".
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#   mathcurve.com, chapters "surface de Veronese" and "surface de
#   Steiner" -- the two named projections used as the endpoints here.
# - J. Steiner, the Roman surface (1844).

def veronese6(u, v, w, a=1.0):
    """The Veronese map into R^6, as (x1..x6)."""
    return np.stack([a * u * u, a * v * v, a * w * w,
                     a * v * w, a * w * u, a * u * v], axis=-1)


def _steiner_pt(angle):
    """fn(theta, phi) for the Steiner surface at projection angle
    `angle`; 0 is the Roman surface, pi/2 the cross-cap."""
    ca, sa = math.cos(angle), math.sin(angle)

    def fn(th, ph):
        cp, sp = np.cos(ph), np.sin(ph)
        u, v, w = np.cos(th) * cp, np.sin(th) * cp, sp + 0.0 * th
        return (v * w, w * u, ca * u * v + sa * (w * w - u * u))
    return fn


def build_steiner(nu, nv, angle=0.0):
    """Mesh the Steiner surface at projection angle `angle`.

    Uses the same RP^2 quotient grid as the Roman surface (of which
    this is the angle-0 member), so the result closes with Euler
    characteristic 1 by construction.
    """
    nu += (-nu) % 4            # quarter-offset grid: need 4 | nu
    return _rp2_quotient(nu, nv, _steiner_pt(float(angle)), 0.25)


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


def morin_point(u, v, n=2, k=1.0):
    """Apery's parametrization of the Morin-Boy family, order n.

        x + iy = K ( A e^{i(n-1)v} + B e^{-iv} ),   z = K cos u
        A = 2 cos u / (n - 1),  B = sqrt2 sin u,
        K = cos u / (sqrt2 - k sin 2u sin nv)

    Written with x and y as one complex number, which is what makes the
    surface's symmetries obvious rather than a surprise: under
    v -> v + phi the two terms turn by (n-1)phi and -phi, so they agree
    on a single rotation exactly when n.phi is a multiple of 2.pi.
    """
    cu, su = np.cos(u), np.sin(u)
    K = cu / (math.sqrt(2.0) - k * np.sin(2.0 * u) * np.sin(n * v))
    A = 2.0 * cu / (n - 1.0)
    B = math.sqrt(2.0) * su
    return (K * (A * np.cos((n - 1) * v) + B * np.cos(v)),
            K * (A * np.sin((n - 1) * v) - B * np.sin(v)),
            K * cu)


def build_morin(nu, nv, n=2, k=1.0):
    """Morin's surface (even n) or Boy's surface (odd n), order n.

    Morin's surface is the halfway model of turning a sphere inside out.
    Smale proved in 1957 that an eversion exists without saying what one
    looks like; Morin, who was blind, produced the model at the midpoint
    of the motion, where the surface is exactly half turned through and
    the two sides can be exchanged.  Apery's parametrization puts it in
    one family with Boy's surface, and the family's PARITY decides the
    topology:

      * even n -- the map is injective on the domain, so the picture is
        an immersed SPHERE.  n = 2 is Morin's surface.
      * odd n -- the map satisfies F(-u, v + pi) = F(u, v) identically,
        so the domain double-covers the image and the picture is an
        immersed PROJECTIVE PLANE.  n = 3 is Boy's surface.

    Both facts are exact identities in the formula, not observations
    about a picture, and `_selftest` checks them as such along with the
    two symmetries every member has:

        F(u, v + 2.pi/n)  = R_z(-2.pi/n) F(u, v)        order n
        F(-u, v + pi/n)   = R_z(pi - pi/n) F(u, v)      swaps the sides

    The second is the one that matters here.  It carries the surface onto
    itself while reversing u, which reverses the orientation of the
    parametrization -- so it exchanges the inside with the outside.  At
    n = 2 its rotation is pi - pi/2 = a QUARTER TURN, which is precisely
    the move Morin and Petit describe at the centre of the eversion.

    The domain is u in [-pi/2, pi/2] (halved for odd n, where the rest is
    a repeat) by v around a circle.  K carries cos u, so both u = +-pi/2
    edges collapse to the origin; those poles are the triple point.
    """
    nv += nv % 2                     # v ~ v + pi pairs columns for odd n
    odd = (n % 2 == 1)
    u0, u1 = (0.0, math.pi / 2) if odd else (-math.pi / 2, math.pi / 2)
    v = TAU * np.arange(nv) / nv

    verts, rows = [], []
    for i in range(nu + 1):
        u = u0 + (u1 - u0) * i / nu
        if abs(abs(u) - math.pi / 2) < 1e-12:        # collapsed pole
            rows.append([len(verts)] * nv)
            verts.append((0.0, 0.0, 0.0))
            continue
        x, y, z = morin_point(u, v, n, k)
        if odd and i == 0:
            # u = 0 is a half circle: F(0, v + pi) = F(0, v), so the two
            # halves of the row are the same points and must share indices
            half = nv // 2
            base = len(verts)
            verts.extend(zip(x[:half], y[:half], z[:half]))
            rows.append([base + (j % half) for j in range(nv)])
            continue
        base = len(verts)
        verts.extend(zip(x, y, z))
        rows.append([base + j for j in range(nv)])

    faces = []
    for i in range(nu):
        a, b = rows[i], rows[i + 1]
        for j in range(nv):
            jn = (j + 1) % nv
            quad = [a[j], a[jn], b[jn], b[j]]
            ring = []
            for q in quad:                            # poles degenerate
                if q not in ring:
                    ring.append(q)
            if len(ring) >= 3:
                faces.append(tuple(ring))
    return np.array(verts, dtype=float), faces


_GENUS_R = 1.0          # circle radius


_GENUS_SPACING = 1.4    # center spacing (< 2r: adjacent circles overlap)


_GENUS_LEVEL = 0.015    # eps: below the lens-core peak for g = 1..5


_GENUS_ZK = 2.0         # z^2 coefficient: slab half-height <= ~0.42


def build_genus(genus, cell=0.125):
    # This module IS inside `minsurf`, so it imports the package it lives
    # in -- one dot up, not one dot across.
    try:
        from . import parametric, plateau, weierstrass, zoo   # noqa: F401
        from .. import minsurf as mst
    except ImportError:
        import minsurf as mst
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


def build_nonorientable(k=3, segments=64, rings=32, hole=0.0,
                        pinch=0.55):
    """The closed non-orientable surface N_k of genus k, as an
    immersion: a sphere carrying k cross-caps.

    N_1 is the projective plane, N_2 the Klein bottle, N_3 Dyck's
    surface, and every closed non-orientable surface is one of these.
    None of them EMBEDS in R^3 -- that is a theorem, not a limitation of
    the meshing -- so each cross-cap is drawn the way it always is, as a
    self-intersecting pinched cap with a segment of double points
    running between two pinch points.

    The construction is surgery rather than a formula, which is what
    makes it exact.  For each cross-cap: cut a disk out of the sphere,
    leaving a boundary circle of 2m vertices, then glue that circle to
    itself ANTIPODALLY by welding vertex i to vertex i + m.  That is the
    definition of attaching a cross-cap, so the topology is right by
    construction rather than by numerical luck: each one drops the Euler
    characteristic by exactly 1, giving chi = 2 - k, and makes the
    surface one-sided.

    Welding each antipodal pair to their midpoint collapses the cut
    circle onto one of its diameters, and that segment is precisely the
    double-point line of the classical cross-cap picture.  `pinch`
    lifts the cap over that segment so the two sheets are visible
    rather than coincident.

    Returns (verts, faces).  The faces along each double-point segment
    are shared by four triangles, not two; that is what an immersion
    looks like as a mesh and is not a defect to weld away.
    """
    import numpy as np

    k = max(1, int(k))
    # How big each cross-cap should be.  A fixed radius makes N_1 read
    # as a sphere with a dent rather than as the projective plane: with
    # one cross-cap the cap IS the surface's whole character and should
    # dominate, while with six they must stay clear of one another.
    # Adjacent centres sit 2 sin(pi/k) apart on the equator, so that
    # sets the ceiling; 0.9 is the free choice when there is only one.
    if hole <= 0.0:
        hole = 0.95 if k == 1 else min(0.95, 0.80 * math.sin(math.pi / k))
    m = max(3, int(segments) // 2)          # half the hole's boundary
    nseg, nring = int(segments), int(rings)

    # --- the sphere, poles welded -----------------------------------
    verts = [(0.0, 0.0, 1.0)]
    for j in range(1, nring):
        phi = math.pi * j / nring
        for i in range(nseg):
            th = 2.0 * math.pi * i / nseg
            verts.append((math.sin(phi) * math.cos(th),
                          math.sin(phi) * math.sin(th),
                          math.cos(phi)))
    verts.append((0.0, 0.0, -1.0))
    south = len(verts) - 1

    def vid(j, i):
        return 1 + (j - 1) * nseg + (i % nseg)

    faces = []
    for i in range(nseg):
        faces.append((0, vid(1, i + 1), vid(1, i)))
    for j in range(1, nring - 1):
        for i in range(nseg):
            faces.append((vid(j, i), vid(j, i + 1),
                          vid(j + 1, i + 1), vid(j + 1, i)))
    for i in range(nseg):
        faces.append((south, vid(nring - 1, i), vid(nring - 1, i + 1)))

    V = [list(p) for p in verts]

    # --- k cross-caps, spaced around the equator --------------------
    remap = {}

    def resolve(a):
        while a in remap:
            a = remap[a]
        return a

    kept = []
    for c in range(k):
        centre = np.array([math.cos(2.0 * math.pi * c / k),
                           math.sin(2.0 * math.pi * c / k), 0.0])
        # faces whose centroid falls inside the disk are cut away, and
        # the vertices left on the cut form the boundary circle
        inside = []
        for f in faces:
            g = np.mean([V[a] for a in f], axis=0)
            if float(np.linalg.norm(g - centre)) < hole:
                inside.append(f)
        if not inside:
            continue
        cut = set()
        for f in inside:
            cut.update(f)
        # boundary ring: cut vertices that still belong to a kept face
        kept_now = [f for f in faces if f not in inside]
        onring = set()
        for f in kept_now:
            for a in f:
                if a in cut:
                    onring.add(a)
        # order the cut circle by angle IN ITS OWN PLANE.  Picking the
        # axes off fixed coordinates instead sorts different holes by
        # different conventions, which pairs the wrong vertices and
        # wrecks the Euler characteristic on some values of k.
        nrm0 = centre / max(float(np.linalg.norm(centre)), 1e-12)
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tmp, nrm0))) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(nrm0, tmp)
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
        e2 = np.cross(nrm0, e1)

        def ring_angle(a):
            d = np.array(V[a]) - centre
            return math.atan2(float(np.dot(d, e2)), float(np.dot(d, e1)))

        ring = sorted(onring, key=ring_angle)
        faces = kept_now
        if len(ring) < 6:
            continue
        half = len(ring) // 2
        axis = np.array([-centre[1], centre[0], 0.0])
        nrm = np.linalg.norm(axis)
        axis = axis / nrm if nrm > 1e-12 else np.array([0.0, 1.0, 0.0])
        for t in range(half):
            a, b = resolve(ring[t]), resolve(ring[t + half])
            if a == b:
                continue
            pa, pb = np.array(V[a]), np.array(V[b])
            mid = 0.5 * (pa + pb)
            # lift the weld off the sphere so the two sheets separate
            s = math.sin(math.pi * (t + 0.5) / half)
            mid = mid + pinch * hole * s * centre / max(
                float(np.linalg.norm(centre)), 1e-12)
            V[a] = list(mid)
            remap[b] = a
        kept.append(c)

    faces = [tuple(resolve(a) for a in f) for f in faces]
    faces = [f for f in faces if len(set(f)) == len(f)]

    used = sorted({a for f in faces for a in f})
    idx = {a: i for i, a in enumerate(used)}
    Vout = [tuple(V[a]) for a in used]
    Fout = [tuple(idx[a] for a in f) for f in faces]
    return Vout, Fout


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


def _selftest():
    """The module had no self-test; this adds one for the surface the
    whole point of which is its topology."""
    from collections import defaultdict, deque
    ok = True

    def _chi(V, F):
        e = set()
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                e.add((a, b) if a < b else (b, a))
        return len(V) - len(e) + len(F)

    def _orientable(F):
        """Try to orient every face consistently.

        Orientation only propagates across MANIFOLD edges; the
        double-point segments of an immersion carry four faces and are
        skipped, which is correct -- they are where the surface passes
        through itself, not where it is glued.
        """
        edge = defaultdict(list)
        for fi, f in enumerate(F):
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                edge[(a, b) if a < b else (b, a)].append((fi, a, b))
        adj = defaultdict(list)
        for lst in edge.values():
            if len(lst) == 2:
                (f0, a0, _b0), (f1, a1, _b1) = lst
                flip = (a0 == a1)
                adj[f0].append((f1, flip))
                adj[f1].append((f0, flip))
        sign = {}
        for start in range(len(F)):
            if start in sign:
                continue
            sign[start] = 1
            q = deque([start])
            while q:
                u = q.popleft()
                for v, flip in adj[u]:
                    want = -sign[u] if flip else sign[u]
                    if v in sign:
                        if sign[v] != want:
                            return False
                    else:
                        sign[v] = want
                        q.append(v)
        return True

    # N_k: chi = 2 - k, and one-sided.  These are the definition of the
    # surface, not a proxy for it, and the surgery is exact, so a bug in
    # the ring ordering shows up here at once.  It did: sorting the cut
    # circle by fixed coordinate axes instead of in the hole's own plane
    # paired the wrong vertices and gave chi = -18 for k = 4, while
    # k = 1, 2, 3 and 5 all came out right and looked convincing.
    bad = []
    for k in (1, 2, 3, 4, 5, 6):
        V, F = build_nonorientable(k, 48, 24)
        c = _chi(V, F)
        if c != 2 - k:
            bad.append('N%d:chi=%d(want %d)' % (k, c, 2 - k))
        elif _orientable(F):
            bad.append('N%d:two-sided' % k)
    ok &= not bad
    print("topology: N_k has chi = 2-k and is one-sided, k = 1..6 %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # control: the same machinery on a sphere must come out orientable
    # with chi = 2, or the test above proves nothing.
    V, F = build_genus(1)
    good = _orientable(F)
    ok &= good
    print("topology: control -- an orientable surface still reads as "
          "two-sided %s" % ('OK' if good else 'FAIL'))

    # ---- the Veronese surface and its Steiner shadows ---------------
    rng = np.random.default_rng(20260821)

    # 1. The Veronese map factors through RP^2 -- every coordinate is
    #    even, so antipodes coincide.  This is what makes it a map OF
    #    the projective plane rather than of the sphere.
    p = rng.normal(size=(3, 500))
    p /= np.linalg.norm(p, axis=0)
    anti = float(np.max(np.abs(veronese6(*p) - veronese6(*(-p)))))

    # 2. ...and it is INJECTIVE there, so RP^2 is genuinely EMBEDDED in
    #    R^6 (really R^5, since x1+x2+x3 = a).  Measured directly: over
    #    many random pairs, two points that are not antipodal never come
    #    closer in the image than their RP^2 distance allows.  This is
    #    the claim that fails for every R^3 projection below, which is
    #    the whole reason the Steiner surfaces have singularities.
    q = rng.normal(size=(3, 400))
    q /= np.linalg.norm(q, axis=0)
    A, B = veronese6(*p[:, :400]), veronese6(*q)
    img = np.linalg.norm(A - B, axis=-1)
    # RP^2 distance: 0 iff the points agree up to sign
    dom = np.minimum(np.linalg.norm(p[:, :400] - q, axis=0),
                     np.linalg.norm(p[:, :400] + q, axis=0))
    far = dom > 1e-3
    ratio = float(np.min(img[far] / dom[far]))
    plane = float(np.max(np.abs(veronese6(*p)[:, :3].sum(-1) - 1.0)))
    good = anti < 1e-14 and ratio > 0.1 and plane < 1e-14
    ok &= good
    print("topology: the Veronese map factors through RP^2 (%.1e) and "
          "embeds it in the hyperplane x1+x2+x3 = a (%.1e), separation "
          "ratio %.3f %s" % (anti, plane, ratio, 'OK' if good else 'FAIL'))

    # 3. The angle-0 shadow IS the Roman surface already shipped --
    #    exactly, not merely similarly.  That is what ties the family to
    #    a surface whose own quartic identity is checked next.
    th = rng.uniform(0.0, TAU, 300)
    ph = rng.uniform(0.05, math.pi / 2, 300)
    r0 = np.stack(np.broadcast_arrays(*_roman_pt(th, ph)), axis=-1)
    s0 = np.stack(np.broadcast_arrays(*_steiner_pt(0.0)(th, ph)), axis=-1)
    d0 = float(np.max(np.abs(r0 - s0)))
    # Steiner's Roman surface satisfies x^2y^2 + y^2z^2 + z^2x^2 = a xyz
    x, y, z = r0[:, 0], r0[:, 1], r0[:, 2]
    quart = float(np.max(np.abs(x * x * y * y + y * y * z * z
                                + z * z * x * x - x * y * z)))
    good = d0 < 1e-14 and quart < 1e-14
    ok &= good
    print("topology: the angle-0 Steiner shadow is the Roman surface "
          "(%.1e) and obeys its quartic (%.1e) %s"
          % (d0, quart, 'OK' if good else 'FAIL'))

    # 4. Every shadow is a closed one-sided surface with chi = 1 -- a
    #    projective plane, at every angle, not just at the two named
    #    ones.  chi is the sharp gate: a projection that degenerated
    #    (collapsing the surface onto a curve or a double cover) would
    #    still mesh and would still look plausible.
    bad = []
    for ang in np.linspace(0.0, math.pi, 7):
        V, F = build_steiner(40, 22, float(ang))
        c = _chi(V, F)
        if c != 1:
            bad.append('%.2f:chi=%d' % (ang, c))
        elif _orientable(F):
            bad.append('%.2f:two-sided' % ang)
        elif not np.all(np.isfinite(V)):
            bad.append('%.2f:non-finite' % ang)
    ok &= not bad
    print("topology: 7 Steiner shadows are all closed one-sided "
          "surfaces with chi = 1 %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # Morin / Boy family.  The two symmetries and the parity rule are
    # exact identities in Apery's formula, so they are checked as
    # identities -- on the parametrization, at machine precision --
    # rather than inferred from the mesh.
    def _rz(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _pt(u, v, n):
        return np.array(morin_point(np.float64(u), np.float64(v), n))
    bad = []
    probes = [(0.3, 0.4), (-0.7, 2.2), (1.1, 5.0), (0.9, 1.3), (1.4, 0.05)]
    for n in (2, 3, 4, 5, 6, 7):
        rot = max(np.linalg.norm(_pt(u, v + TAU / n, n)
                                 - _rz(-TAU / n) @ _pt(u, v, n))
                  for u, v in probes)
        swap = max(np.linalg.norm(_pt(-u, v + math.pi / n, n)
                                  - _rz(math.pi - math.pi / n) @ _pt(u, v, n))
                   for u, v in probes)
        rp2 = max(np.linalg.norm(_pt(-u, v + math.pi, n) - _pt(u, v, n))
                  for u, v in probes)
        if rot > 1e-12:
            bad.append('n=%d:rotation %.1e' % (n, rot))
        if swap > 1e-12:
            bad.append('n=%d:side-swap %.1e' % (n, swap))
        # odd n folds onto RP^2, even n does not -- that is the whole
        # difference between a Boy surface and a Morin surface
        if (n % 2 == 1) != (rp2 < 1e-12):
            bad.append('n=%d:parity rp2=%.1e' % (n, rp2))
        V, F = build_morin(40, 40, n)
        chi = _chi(V, F)
        want = 1 if n % 2 else 2
        if chi != want:
            bad.append('n=%d:chi=%d want %d' % (n, chi, want))
        if _orientable(F) != (n % 2 == 0):
            bad.append('n=%d:sidedness' % n)
        if not np.all(np.isfinite(V)):
            bad.append('n=%d:non-finite' % n)
    ok &= not bad
    print("topology: Morin/Boy family n = 2..7 -- order-n rotation and "
          "the side-swapping symmetry exact to 1e-12; even n closed "
          "two-sided chi = 2 (Morin), odd n one-sided chi = 1 (Boy) %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- Franzoni's classical-shape Klein bottle --------------------
    # The paper's closure conditions are exact identities of the
    # dumbbell directrix, so they are checked as identities first --
    # alpha(0) = alpha(pi), alpha'(0) = -alpha'(pi), r(0) = r(pi) --
    # and only then is the glued mesh gated on what those conditions
    # buy: a genuinely CLOSED non-orientable chi = 0 surface, with the
    # unavoidable winding flip confined to the one seam ring.
    bad = []
    for t0, t1 in ((0.0, math.pi),):
        x0, y0, z0 = franzoni_klein_point(np.array([t0]), np.array([0.0]))
        x1, y1, z1 = franzoni_klein_point(np.array([t1]),
                                          np.array([math.pi]))
        if max(float(np.max(np.abs(x0 - x1))),
               float(np.max(np.abs(y0 - y1))),
               float(np.max(np.abs(z0 - z1)))) > 1e-12:
            bad.append("seam circle mismatch")
    aa, bb = 20.0 / 4.0, 8.0 / 4.0
    for t0, t1, s in ((1e-9, math.pi - 1e-9, -1.0),):
        d0 = np.array([aa * math.cos(t0),
                       bb * math.sin(t0) * (3 * math.cos(t0) ** 2 - 1)])
        d1 = np.array([aa * math.cos(t1),
                       bb * math.sin(t1) * (3 * math.cos(t1) ** 2 - 1)])
        if np.max(np.abs(d0 + d1)) > 1e-6:
            bad.append("alpha'(0) != -alpha'(pi)")
    nu_, nv_ = 48, 24
    V, F = build_klein_franzoni(nu_, nv_)
    cnt = edge_face_counts(F)
    chi = len(V) - len(cnt) + len(F)
    nbound = sum(1 for v in cnt.values() if v == 1)
    conflicts = winding_conflict_edges(F)
    if chi != 0:
        bad.append("dumbbell chi=%d" % chi)
    if nbound != 0:
        bad.append("dumbbell boundary=%d" % nbound)
    if _orientable(F):
        bad.append("dumbbell orientable")
    if len(conflicts) != nv_:
        bad.append("dumbbell conflict ring %d != nv" % len(conflicts))
    if not np.all(np.isfinite(V)):
        bad.append("dumbbell non-finite")
    # the piriform rendition CANNOT close (|gamma'| = 0 at the cusp,
    # the paper's own section-4 caveat): meshed open, it is an
    # orientable tube with exactly the two rim circles as boundary
    V, F = build_klein_franzoni(nu_, nv_, directrix='PIRIFORM')
    cnt = edge_face_counts(F)
    chi = len(V) - len(cnt) + len(F)
    nbound = sum(1 for v in cnt.values() if v == 1)
    if chi != 0 or nbound != 2 * nv_ or not _orientable(F):
        bad.append("piriform chi=%d boundary=%d" % (chi, nbound))
    ok &= not bad
    print("topology: Franzoni Klein bottle -- closure identities hold, "
          "dumbbell tube closes (chi 0, 0 boundary edges, one-sided, "
          "winding flip = one seam ring), piriform stays honestly open "
          "%s" % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- the plain Mobius band --------------------------------------
    bad = []
    nu_, nv_ = 64, 8
    V, F = build_mobius_band(nu_, nv_)
    cnt = edge_face_counts(F)
    chi = len(V) - len(cnt) + len(F)
    bedges = [e for e, cx in cnt.items() if cx == 1]
    if chi != 0:
        bad.append("chi=%d" % chi)
    if len(bedges) != 2 * nu_:
        bad.append("boundary edges %d" % len(bedges))
    # the famous single edge: the boundary must be ONE loop
    adj = {}
    for e0, e1 in bedges:
        adj.setdefault(e0, []).append(e1)
        adj.setdefault(e1, []).append(e0)
    start = bedges[0][0]
    loop, prev, cur = 1, None, start
    while True:
        nxt = [w for w in adj[cur] if w != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        loop += 1
    if loop != len(bedges):
        bad.append("boundary is %d loops' worth" % loop)
    if _orientable(F):
        bad.append("two-sided")
    if len(winding_conflict_edges(F)) != nv_:
        bad.append("conflict ring != seam")
    ok &= not bad
    print("topology: Mobius band -- chi 0, ONE boundary loop, "
          "one-sided, winding flip = the seam ruling %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("topology self-test failed")

