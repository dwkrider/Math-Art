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
# - W. von Dyck, "Beitraege zur Analysis situs", Math. Ann. 32 (1888),
#   457-512 -- the classification insight behind Dyck's surface: a
#   sphere with three cross-caps is the same closed surface as a torus
#   with one.
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#   mathcurve.com, chapter "surface de Dyck" -- both forms of the
#   surface, and Christoph Soland's octagon presentation (1 face,
#   4 edges, 2 vertices) realized in his wire sculpture "Janus
#   bifrons" (Gymnase du Bugnon, Lausanne).
# - F. Klein, "Ueber die Transformation siebenter Ordnung der
#   elliptischen Functionen", Math. Ann. 14 (1878) -- the quartic
#   curve and its regular map {3,7}_8 of genus 3, whose 168
#   orientation-preserving symmetries are the most a genus-3 surface
#   allows.
# - E. Schulte and J. M. Wills, "A polyhedral realization of Felix
#   Klein's map {3,7}_8 on a Riemann surface of genus 3", J. London
#   Math. Soc. (2) 32 (1985), 539-547 -- the embedded 56-triangle
#   polyhedron on two homothetic truncated tetrahedra used here.
# - S. Levy (ed.), "The Eightfold Way: The Beauty of Klein's Quartic
#   Curve", MSRI Publications 35, Cambridge University Press (1999) --
#   the volume around Helaman Ferguson's sculpture of the 24-heptagon
#   tiling.
# - J. C. Baez, "Klein's Quartic Curve",
#   math.ucr.edu/home/baez/klein.html -- the 336-fold symmetry and the
#   {3,7} / {7,3} tilings, plainly told.
# - G. Egan, "Klein's Quartic Curve",
#   gregegan.net/SCIENCE/KleinQuartic/KleinQuartic.html -- tetrahedral
#   realizations of the tilings and the dual heptagonal view.
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
    """The polynomial bottle-shaped Klein immersion (u in [0, pi],
    v in [0, 2pi]).  The u = pi rim coincides with the u = 0 rim under
    v -> pi - v, i.e. column j of the last row is column nv/2 - j of the
    first (verified to 5e-16), and that identification is APPLIED here:
    the last row is not emitted at all, its face corners referring back
    into row 0.  So the mesh is genuinely closed -- chi = 0 with no
    boundary.

    This used to leave the seam SPLIT, as coincident duplicate vertices
    with no index gluing, because welding makes the winding flip there
    and averaged smooth normals degenerate into a dark crease.  That
    bought smooth shading at the price of a "closed" surface with 96
    boundary edges, which is not closed.  The honest fix is the one the
    Franzoni rendition uses: close the mesh and mark the one
    unavoidable winding-conflict ring SHARP, which splits normals at
    exactly that ring (see `winding_conflict_edges` and the operator's
    seam handling).  Every closed non-orientable mesh has such a ring;
    it is a fact about non-orientability, not a defect to hide by
    leaving a hole."""
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
    V = V[:nu * nv]                      # drop the duplicate u = pi row

    def idx(i, j):
        """Grid index, folding the last row onto the first.

        The seam identification is v -> pi - v, which on the sample grid
        v_j = 2 pi j / nv is j -> nv/2 - j.  nv is forced even above so
        that lands on a sample.
        """
        if i == nu:
            return ((nv // 2 - j) % nv)
        return i * nv + (j % nv)

    faces = []
    for i in range(nu):
        for j in range(nv):
            j2 = j + 1
            faces.append((idx(i, j), idx(i, j2),
                          idx(i + 1, j2), idx(i + 1, j)))
    return V, faces


def build_klein_figure8(nu, nv, radius=2.0):
    """Figure-8 (twisted-torus) Klein immersion: the cross-section is a
    figure-8 that makes a half-turn per revolution.  v samples sit at
    half-steps so no column lands on the figure-8 crossing point.

    The u = 2pi seam coincides with u = 0 under v -> -v, and because of
    those half-steps that is j -> nv - 1 - j on the sample grid, NOT the
    j -> -j the continuous map suggests (verified to 2e-15; the naive
    map is wrong by 0.29).  The identification is applied, so the mesh
    is closed: chi = 0 with no boundary.  It used to be left split, and
    the one unavoidable winding-conflict ring of a closed non-orientable
    mesh is marked sharp by the operator instead -- see
    build_klein_bottle."""
    u = TAU * np.arange(nu + 1)[:, None] / nu
    v = TAU * (np.arange(nv)[None, :] + 0.5) / nv
    c2, s2 = np.cos(u / 2), np.sin(u / 2)
    sv, s2v = np.sin(v), np.sin(2 * v)
    r = radius + c2 * sv - s2 * s2v
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = s2 * sv + c2 * s2v
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    V = V[:nu * nv]                      # drop the duplicate u = 2pi row

    def idx(i, j):
        j %= nv
        return (nv - 1 - j) if i == nu else i * nv + j

    faces = []
    for i in range(nu):
        for j in range(nv):
            faces.append((idx(i, j), idx(i, j + 1),
                          idx(i + 1, j + 1), idx(i + 1, j)))
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


def ovalesque_point(th, ta, l, b, r1=1.0, r2=2.0):
    """Francis's ovalesque sweep F(l, b) (A Topological Picturebook,
    pp. 96, 178-179): the plane quartic

        rho(tau) = ((1 - l) cos tau + l)
                   / (1 - (b/sqrt 2) sin(3 theta) sin(2 tau))

    carried into space by the affine maps L(theta) = <J, K> with
    J = (r1 cos 2 theta, r1 sin 2 theta, r2) (altitudinal axis) and
    K = (cos theta, -sin theta, 0) (basal axis):

        P = rho cos(tau) J(theta) + rho sin(tau) K(theta).

    Corners of the family: F(0,0) is Apery's cylindrical Roman
    surface, F(0,1) IS Apery's published Boy immersion (exactly, with
    r1 = 1/sqrt 2, r2 = 3/2 -- the self-test checks this to 1e-12),
    F(1,0) the ETRUSCAN VENUS (a singular Klein bottle: the connected
    sum of two Roman surfaces, with 12 pinch points), F(1,1) IDA (a
    smooth immersed Klein bottle).  The closure identity
    P(theta + pi, -tau) = P(theta, tau) glues the mesh; for l = 0 the
    additional symmetry P(theta, tau + pi) = P(theta, tau) makes the
    image a double-covered projective plane instead."""
    den = 1.0 - (b / math.sqrt(2.0)) * np.sin(3.0 * th)         * np.sin(2.0 * ta)
    rho = ((1.0 - l) * np.cos(ta) + l) / den
    A = rho * np.cos(ta)
    B = rho * np.sin(ta)
    return np.stack([A * r1 * np.cos(2.0 * th) + B * np.cos(th),
                     A * r1 * np.sin(2.0 * th) - B * np.sin(th),
                     A * r2], axis=-1)


def build_ovalesque(nu, nv, l, b, r1=1.0, r2=2.0):
    """Mesh F(l, b) on the closed (Klein) domain: theta runs over
    [0, pi) and tau over [0, 2 pi), and the theta = pi row is glued to
    theta = 0 under tau -> -tau (the closure identity above), so the
    mesh is CLOSED with chi = 0.  For l = 1 (Venus, Ida) it is
    one-sided -- a Klein bottle; the self-test measures that, and the
    pinch-point distinction between the singular Venus and the
    immersed Ida.  Returns (verts, faces)."""
    nu = max(8, int(nu))
    nv = max(8, int(nv))
    th = math.pi * np.arange(nu)[:, None] / nu
    ta = TAU * np.arange(nv)[None, :] / nv
    V = ovalesque_point(th, ta, float(l), float(b), r1, r2)
    verts = [tuple(v) for v in V.reshape(-1, 3)]

    def vid(i, j):
        if i < nu:
            return i * nv + j % nv
        return (nv - j) % nv          # theta = pi ~ theta = 0, tau -> -tau
    faces = []
    for i in range(nu):
        for j in range(nv):
            q = (vid(i, j), vid(i, j + 1),
                 vid(i + 1, j + 1), vid(i + 1, j))
            if len(set(q)) == 4:
                faces.append(q)
    return verts, faces


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


def attach_crosscaps(verts, faces, centres, hole, pinch):
    """The cross-cap surgery, on any closed mesh: for each centre, cut
    away the faces whose centroid lies within `hole` of it and close
    the boundary circle with a MOBIUS BAND -- which is what a
    projective plane minus a disk is, so gluing one into a circle is
    the definition of attaching a cross-cap.  Each cap therefore drops
    the Euler characteristic by exactly 1 and makes the surface
    one-sided, by construction rather than by numerical luck.

    The band is meshed on the classical cross-cap immersion
    K(x, y, z) = (yz, xy, (x^2 - y^2)/2) of the sphere (the same
    surface the CROSSCAP preset draws; every coordinate is even, so
    antipodes coincide and the band's core row welds vertex t to
    vertex t + m exactly).  The sphere is cut at colatitude pi/4
    around the smooth point opposite the pinch, and the patch is
    seated on the rim with the double segment pointing along the
    centre direction: the visible cap is a dome that rises off the
    surface and passes through itself along a SEGMENT of double
    points terminating in two pinch points (Whitney umbrellas) -- the
    textbook cross-cap picture.  An earlier version instead welded
    the boundary ring itself to antipodal midpoints; every such
    midpoint sits at the hole's centre, so the whole cap collapsed
    onto its lifted spine and rendered as a fan of slivers.  The
    self-test now gates on the cap's height AND width, which is what
    would have caught that.

    The boundary ring is walked edge by edge in the direction induced
    by the kept faces, and each rim vertex keeps the angle it makes IN
    THE HOLE'S OWN PLANE (normal = the centre direction), which is
    what seats the band's boundary on the rim.  Measuring angles off
    fixed coordinate axes instead treats different holes by different
    conventions and pairs the wrong vertices -- that bug shipped once,
    and the N_k self-test is what caught it.

    `pinch` scales the cap's height along the centre direction (1 is
    the immersion's own proportion, 0 flattens the cap).  Returns
    (verts, faces), compacted to the used vertices.
    """
    V = [list(p) for p in verts]
    faces = [tuple(f) for f in faces]

    def _boundary_loop(kept):
        """The cut's rim as one vertex cycle, walked in the direction
        the kept faces traverse it; None if the rim is pinched or not
        a single loop."""
        cnt = {}
        for f in kept:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                e = (a, b) if a < b else (b, a)
                cnt[e] = cnt.get(e, 0) + 1
        succ = {}
        for f in kept:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                e = (a, b) if a < b else (b, a)
                if cnt[e] == 1:
                    if a in succ:
                        return None
                    succ[a] = b
        if not succ:
            return None
        start = next(iter(succ))
        loop = [start]
        x = succ[start]
        while x != start:
            loop.append(x)
            x = succ.get(x)
            if x is None or len(loop) > len(succ):
                return None
        return loop if len(loop) == len(succ) else None

    for centre in centres:
        centre = np.asarray(centre, dtype=float)
        # faces whose centroid falls inside the disk are cut away
        cid = [np.mean([V[a] for a in f], axis=0) for f in faces]
        inside = {fi for fi, g in enumerate(cid)
                  if float(np.linalg.norm(g - centre)) < hole}
        if not inside:
            continue
        # the rim must be one loop of even length (the antipodal weld
        # needs a partner for every vertex; a contractible cycle in a
        # quad grid is always even, so growth is a rare fallback):
        # grow the cut by the nearest touching face until it is
        ring = None
        for _ in range(8):
            kept = [f for fi, f in enumerate(faces)
                    if fi not in inside]
            ring = _boundary_loop(kept)
            if (ring is not None and len(ring) >= 6
                    and len(ring) % 2 == 0):
                break
            cutverts = set()
            for fi in inside:
                cutverts.update(faces[fi])
            grow = [(float(np.linalg.norm(cid[fi] - centre)), fi)
                    for fi in range(len(faces))
                    if fi not in inside and cutverts & set(faces[fi])]
            if not grow:
                break
            inside.add(min(grow)[1])
            ring = None
        if ring is None or len(ring) < 6 or len(ring) % 2:
            raise ValueError("cross-cap cut has no clean even rim")
        faces = kept
        L = len(ring)
        m = L // 2

        nrm0 = centre / max(float(np.linalg.norm(centre)), 1e-12)
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tmp, nrm0))) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(nrm0, tmp)
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
        e2 = np.cross(nrm0, e1)
        pts = np.array([V[a] for a in ring])
        base = pts.mean(axis=0)
        d = pts - centre
        th = np.arctan2(d @ e2, d @ e1)
        dip = d - np.outer(d @ nrm0, nrm0)
        rho = float(np.mean(np.linalg.norm(dip, axis=1)))
        # The immersion's boundary circle is not planar -- its height
        # varies with the SECOND harmonic of the azimuth -- and
        # neither is the cut rim on a curved host.  Twist the cap
        # about its axis so the two wobbles line up in phase (fit the
        # rim heights' second harmonic); with an arbitrary twist the
        # residuals alternate around the ring and the blended seam
        # ripples visibly.
        hgt = (pts - base) @ nrm0
        phase = 0.5 * math.atan2(float(np.sum(hgt * np.sin(2 * th))),
                                 float(np.sum(hgt * np.cos(2 * th))))
        delta = phase + math.pi / 2.0
        cd, sd = math.cos(delta), math.sin(delta)
        alpha = -th - math.pi / 2.0 + delta  # rim angle -> sphere angle
        # Cut colatitude: how much of the cross-cap blob the cap
        # keeps.  pi/4 would seat the blob's widest circle on the rim
        # and bury its lower pinch point inside the host surface;
        # cutting the much smaller disk keeps nearly the whole blob,
        # so the cap sits on the hole as a rounded ball -- belly wider
        # than its neck -- with the double segment and both pinch
        # points clear of the surface, the way the classical pictures
        # (e.g. mathcurve's Dyck renderings) draw it.
        b0 = math.pi / 8.0
        scale = rho / (math.sin(b0) * math.cos(b0))
        nrows = max(6, m)

        def kpt(al, be):
            """Cross-cap immersion of the sphere point at colatitude
            `be` from the cut centre, azimuth `al`."""
            sb, cb = math.sin(be), math.cos(be)
            x, y, z = sb * math.cos(al), -cb, sb * math.sin(al)
            return np.array([y * z, x * y, 0.5 * (x * x - y * y)])

        rim0 = [kpt(alpha[i], b0) for i in range(L)]
        z0 = float(np.mean([q[2] for q in rim0]))

        def world(q):
            # the twist rotates the in-plane components back so the
            # boundary still seats on the rim at its own angle
            qx = q[0] * cd - q[1] * sd
            qy = q[0] * sd + q[1] * cd
            return (base + scale * (qx * e1 + qy * e2)
                    + scale * pinch * (q[2] - z0) * nrm0)

        # rows from the rim (kept verbatim) up to the antipodally
        # welded core, which lands on the double segment: m vertices
        # shared by both sheets by INDEX for the weld, while the
        # sheets crossing there stay separate faces.  The ideal rim's
        # residual against the true rim is blended away over the first
        # rows -- but only its SMOOTH part (harmonics 0..2 in the rim
        # angle: offset, tilt and ellipticity).  The raw residual also
        # carries the staircase of the cut through the host's quad
        # grid, and blending that in scallops the whole dome with
        # grid-frequency wrinkles; fitted, the staircase stays where
        # it belongs, in the one quad ring at the seam.
        resid = pts - np.array([world(q) for q in rim0])
        H = np.stack([np.ones(L), np.cos(th), np.sin(th),
                      np.cos(2.0 * th), np.sin(2.0 * th)], axis=1)
        coef, _r, _rk, _sv = np.linalg.lstsq(H, resid, rcond=None)
        offs = H @ coef
        rows = [list(ring)]
        for j in range(1, nrows):
            be = b0 + (math.pi / 2.0 - b0) * j / nrows
            w = 0.5 * (1.0 + math.cos(math.pi * j / nrows))
            row = []
            for i in range(L):
                p = world(kpt(alpha[i], be)) + w * offs[i]
                row.append(len(V))
                V.append(list(p))
            rows.append(row)
        core = []
        for t in range(m):
            p = world(kpt(alpha[t], math.pi / 2.0))
            core.append(len(V))
            V.append(list(p))
        rows.append(core)
        for j in range(nrows):
            lo, hi = rows[j], rows[j + 1]
            nh = len(hi)
            for t in range(L):
                q = (lo[(t + 1) % L], lo[t],
                     hi[t % nh], hi[(t + 1) % L % nh])
                if len(set(q)) == len(q):
                    faces.append(q)

    faces = [f for f in faces if len(set(f)) == len(f)]
    used = sorted({a for f in faces for a in f})
    idx = {a: i for i, a in enumerate(used)}
    return ([tuple(V[a]) for a in used],
            [tuple(idx[a] for a in f) for f in faces])


def build_nonorientable(k=3, segments=64, rings=32, hole=0.0,
                        pinch=1.0):
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
    leaving a boundary circle of 2m vertices, then close it with a
    Mobius band -- a projective plane minus a disk -- whose core row
    is glued to itself ANTIPODALLY, vertex t welded to vertex t + m.
    That is the definition of attaching a cross-cap, so the topology
    is right by construction rather than by numerical luck: each one
    drops the Euler characteristic by exactly 1, giving chi = 2 - k,
    and makes the surface one-sided.

    The band is shaped on the classical cross-cap immersion, so each
    cap is a dome rising off the sphere whose two sheets cross along
    a segment of double points between two pinch points.  `pinch`
    scales the dome's height (1 is the immersion's own proportion).

    Returns (verts, faces).  The surgery itself lives in
    `attach_crosscaps`, shared with `build_dyck`.
    """
    k = max(1, int(k))
    # How big each cross-cap should be.  A fixed radius makes N_1 read
    # as a sphere with a dent rather than as the projective plane: with
    # one cross-cap the cap IS the surface's whole character and should
    # dominate, while with six they must stay clear of one another.
    # Adjacent centres sit 2 sin(pi/k) apart on the equator, and the
    # cap's belly bulges well past its cut circle now that the caps
    # are full cross-cap blobs, so the per-cap budget is tighter than
    # it was for the old flat welds; 0.75 is the free choice when
    # there is only one cap.
    if hole <= 0.0:
        hole = 0.75 if k == 1 else min(0.75, 0.55 * math.sin(math.pi / k))
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

    # --- k cross-caps, spaced around the equator --------------------
    centres = [(math.cos(TAU * c / k), math.sin(TAU * c / k), 0.0)
               for c in range(k)]
    return attach_crosscaps(verts, faces, centres, hole, pinch)


def build_dyck(segments=64, rings=32, hole=0.0, pinch=1.0,
               major=1.0, minor=0.45):
    """Dyck's surface as a TORUS carrying one cross-cap.

    Von Dyck proved in 1888 that sewing three cross-caps into a sphere
    gives the same closed surface as sewing ONE cross-cap into a torus:
    P^2 # P^2 # P^2 = T^2 # P^2, the relation that collapses the
    classification of surfaces down to the two familiar families.  The
    surface with those two forms has been called Dyck's surface since.
    It is the closed non-orientable surface of genus 3 -- one-sided,
    Euler characteristic -1 -- and like every closed one-sided surface
    it cannot embed in 3-space, so the cross-cap is drawn the usual
    way, as a pinched cap with a segment of double points.

    `build_nonorientable(3)` is the left-hand form (the sphere with
    three cross-caps).  This builder is the right-hand form: a torus
    with a single cross-cap grafted onto its outer equator, which reads
    completely differently -- a handle AND a cross-cap -- and is the
    form that makes von Dyck's theorem worth a picture.  The same
    exact surgery is used (`attach_crosscaps`): chi(torus) = 0, and
    the one cap drops it by exactly 1.

    Returns (verts, faces).
    """
    nseg, nring = max(16, int(segments)), max(8, int(rings))
    R, r = float(major), float(minor)
    if hole <= 0.0:
        # a cut of ~0.6 tube radii grows into a cap whose belly is
        # about the tube's own girth -- the proportion of the
        # classical torus-with-cross-cap renderings
        hole = 0.60 * r
    # the tube angle is sampled at half-steps so no vertex row lands
    # exactly on the outer equator the cap is centred on
    verts = []
    for i in range(nseg):
        u = TAU * i / nseg
        cu, su = math.cos(u), math.sin(u)
        for j in range(nring):
            v = TAU * (j + 0.5) / nring
            w = R + r * math.cos(v)
            verts.append((w * cu, w * su, r * math.sin(v)))
    faces = []
    for i in range(nseg):
        i2 = (i + 1) % nseg
        for j in range(nring):
            j2 = (j + 1) % nring
            faces.append((i * nring + j, i2 * nring + j,
                          i2 * nring + j2, i * nring + j2))
    return attach_crosscaps(verts, faces, [(R + r, 0.0, 0.0)],
                            hole, pinch)


# ----------------------------------------------------------------------
# the Klein quartic
# ----------------------------------------------------------------------
# Klein's quartic curve x^3 y + y^3 z + z^3 x = 0 lives in the COMPLEX
# projective plane, so the equation cannot be meshed directly; what can
# be realized in 3-space is the genus-3 surface it defines, carrying
# the combinatorial structure that makes it famous: the regular map
# {3,7}_8 -- 56 triangles, 84 edges, 24 vertices, seven triangles
# around every vertex, Petrie polygons of length 8 -- whose
# automorphism group PSL(2,7) has order 168 (336 with reflections),
# the maximum 84(g-1) that Hurwitz allows a genus-3 surface.
#
# The realization used here is Schulte and Wills's polyhedron (1985):
# an EMBEDDED polyhedron of genus 3 with 56 flat triangular faces whose
# 24 vertices are two homothetic truncated tetrahedra -- the even-sign-
# change permutations of (1, 1, 3), together with the same twelve
# points scaled by 1/2 -- with the full combinatorial symmetry of
# Klein's map (the self-test counts all 336 automorphisms on the mesh)
# though only the 12 tetrahedral rotations act as rigid motions.  The
# dual view, computed from it face-by-face, is the {7,3} tiling by 24
# heptagons, three around each corner: the tiling of Helaman Ferguson's
# sculpture "The Eightfold Way" (MSRI, 1993).

_KLEIN_MAP_VERTS = (
    (1, 1, 3), (1, -3, -1), (-3, 1, -1), (1, 3, 1),
    (-1.5, -0.5, 0.5), (-1, -1, 3), (1.5, 0.5, 0.5),
    (1.5, -0.5, -0.5), (0.5, -1.5, -0.5), (3, 1, 1),
    (1, -1, -3), (-0.5, -0.5, 1.5), (0.5, -0.5, -1.5),
    (-1.5, 0.5, -0.5), (3, -1, -1), (-1, -3, 1),
    (-0.5, 0.5, -1.5), (-3, -1, 1), (0.5, 0.5, 1.5),
    (-0.5, -1.5, 0.5), (0.5, 1.5, 0.5), (-1, 1, -3),
    (-0.5, 1.5, -0.5), (-1, 3, -1))

_KLEIN_MAP_TRIS = (
    (0, 9, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6), (0, 6, 7),
    (0, 7, 8), (0, 8, 9), (1, 13, 10), (1, 14, 11),
    (1, 15, 12), (1, 16, 13), (1, 10, 14), (1, 11, 15),
    (1, 12, 16), (2, 19, 17), (2, 20, 18), (2, 21, 19),
    (2, 22, 20), (2, 23, 21), (2, 17, 22), (2, 18, 23),
    (3, 11, 4), (4, 13, 5), (5, 15, 6), (6, 10, 7),
    (7, 12, 8), (8, 14, 9), (9, 16, 3), (3, 18, 11),
    (4, 21, 13), (5, 17, 15), (6, 20, 10), (7, 23, 12),
    (8, 19, 14), (9, 22, 16), (3, 23, 18), (4, 19, 21),
    (5, 22, 17), (6, 18, 20), (7, 21, 23), (8, 17, 19),
    (9, 20, 22), (3, 16, 23), (4, 11, 19), (5, 13, 22),
    (6, 15, 18), (7, 10, 21), (8, 12, 17), (9, 14, 20),
    (10, 20, 14), (11, 18, 15), (12, 23, 16), (13, 21, 10),
    (14, 19, 11), (15, 17, 12), (16, 22, 13))


def build_klein_quartic(dual=True):
    """The Klein quartic's genus-3 surface with its regular tiling
    carried through to the mesh (see the section comment above).

    dual=False gives the {3,7} side: the Schulte-Wills polyhedron
    itself, 56 flat triangles with seven around every vertex.
    dual=True (the default) gives the {7,3} side computed from it: 24
    heptagons, three around every corner, each heptagon the ring of
    face centroids around one primal vertex -- the "Eightfold Way"
    view.  The heptagons are not planar (they cannot be, on a genus-3
    polyhedron), which is fine for a mesh.

    Both views are closed, orientable, chi = -4, and carry the full
    336-element combinatorial automorphism group of Klein's map, which
    the self-test verifies flag by flag.  Returns (verts, faces),
    consistently wound.
    """
    V = [tuple(float(c) for c in p) for p in _KLEIN_MAP_VERTS]
    F = [tuple(f) for f in _KLEIN_MAP_TRIS]
    if not dual:
        return V, F
    # dual vertices: primal face centroids
    DV = [tuple(sum(V[a][c] for a in f) / len(f) for c in range(3))
          for f in F]
    # dual faces: the faces around each primal vertex, walked in
    # winding order (in face fi the edge v -> next(v) is shared with
    # exactly one other face; stepping across it circulates around v)
    ef = {}
    nxt = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            ef.setdefault((a, b) if a < b else (b, a), []).append(fi)
            nxt[(fi, a)] = b
    first = {}
    count = {}
    for fi, f in enumerate(F):
        for v in f:
            first.setdefault(v, fi)
            count[v] = count.get(v, 0) + 1
    DF = []
    for v in range(len(V)):
        start = first[v]
        cyc = [start]
        fi = start
        while True:
            w = nxt[(fi, v)]
            fs = ef[(v, w) if v < w else (w, v)]
            fj = fs[0] if fs[1] == fi else fs[1]
            if fj == start:
                break
            cyc.append(fj)
            fi = fj
        if len(cyc) != count[v]:
            raise ValueError("dual walk did not close at vertex %d" % v)
        DF.append(tuple(cyc))
    return DV, DF


def _flag_system(F):
    """The flag system of a closed polygonal mesh: every incident
    (vertex, edge, face) triple, with the three adjacency involutions
    s0 (other vertex of the edge), s1 (other edge of the face at that
    vertex) and s2 (other face on the edge).  Returns (flags, s0, s1,
    s2): the flag list as (vertex, edge, face) triples and the three
    involutions as permutation lists.  Combinatorial map automorphisms
    commute with the involutions, which is what makes the flag system
    the right instrument for counting them."""
    ef = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            ef.setdefault((a, b) if a < b else (b, a), []).append(fi)
    flags = []
    fid = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            e = (a, b) if a < b else (b, a)
            for v in (a, b):
                fid[(v, e, fi)] = len(flags)
                flags.append((v, e, fi))
    vedges = {}
    for fi, f in enumerate(F):
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            e = (a, b) if a < b else (b, a)
            vedges.setdefault((fi, a), []).append(e)
            vedges.setdefault((fi, b), []).append(e)
    s0, s1, s2 = [], [], []
    for v, e, fi in flags:
        s0.append(fid[(e[0] if e[1] == v else e[1], e, fi)])
        other = [ee for ee in vedges[(fi, v)] if ee != e]
        s1.append(fid[(v, other[0], fi)])
        fs = ef[e]
        s2.append(fid[(v, e, fs[0] if fs[1] == fi else fs[1])])
    return flags, s0, s1, s2


def _map_automorphism_count(F):
    """How many combinatorial automorphisms the mesh has AS A MAP,
    counted directly: an automorphism is determined by where it sends
    one flag, so try every target flag and propagate through the three
    involutions, counting the targets where the propagation closes
    without conflict.  A regular map -- one whose symmetry group acts
    transitively on flags -- scores exactly its flag count."""
    from collections import deque
    flags, s0, s1, s2 = _flag_system(F)
    n = len(flags)
    S = (s0, s1, s2)
    good = 0
    for target in range(n):
        m = {0: target}
        q = deque([0])
        ok = True
        while q and ok:
            x = q.popleft()
            y = m[x]
            for s in S:
                x2, y2 = s[x], s[y]
                got = m.get(x2)
                if got is None:
                    m[x2] = y2
                    q.append(x2)
                elif got != y2:
                    ok = False
                    break
        if ok and len(m) == n:
            good += 1
    return good


def _map_petrie_lengths(F):
    """The set of Petrie polygon lengths of the mesh as a map: orbit
    lengths of the composed flag step s2 s1 s0 (one zigzag stride).
    A cube scores {6} -- its Petrie hexagons -- and Klein's map {3,7}_8
    scores {8}, the subscript in its name."""
    flags, s0, s1, s2 = _flag_system(F)
    n = len(flags)
    lens = set()
    seen = [False] * n
    for t in range(n):
        if seen[t]:
            continue
        x = t
        k = 0
        while True:
            seen[x] = True
            x = s2[s1[s0[x]]]
            k += 1
            if x == t:
                break
        lens.add(k)
    return lens


def subdivide_flags(V, F):
    """Barycentric subdivision of a closed polygonal mesh into its
    FLAGS: each k-gon splits into 2k right-ish triangles (vertex, edge
    midpoint, face centroid), one per incident (vertex, edge, face)
    triple.  Klein drew his 1879 figure of the quartic exactly this
    way -- 24 heptagons each cut into 14 little triangles -- because
    the 24 x 14 = 336 triangles then stand one-for-one for the map's
    336 symmetries including reflections: a symmetry is determined by
    where it sends a single flag.

    Returns (verts, tris, info) with info[i] = (v, e, f, c) for
    triangle i: the flag's vertex index, its edge as a sorted vertex
    pair, its face index in F, and its handedness c (0 or 1).  The
    handedness classes checkerboard the subdivision -- triangles
    sharing an edge always differ -- and on Klein's map they split
    336 = 168 + 168: the orientation-preserving and the
    orientation-reversing halves of the symmetry group, made visible.
    """
    V2 = [tuple(float(c) for c in p) for p in V]
    emid = {}
    for f in F:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            e = (a, b) if a < b else (b, a)
            if e not in emid:
                emid[e] = len(V2)
                V2.append(tuple(
                    0.5 * (np.asarray(V2[a]) + np.asarray(V2[b]))))
    tris, info = [], []
    for fi, f in enumerate(F):
        c = np.mean([V2[a] for a in f], axis=0)
        ci = len(V2)
        V2.append(tuple(float(x) for x in c))
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            e = (a, b) if a < b else (b, a)
            mi = emid[e]
            tris.append((a, mi, ci))
            info.append((a, e, fi, 0))
            tris.append((mi, b, ci))
            info.append((b, e, fi, 1))
    return V2, tris, info


def klein_quartic_rotations():
    """The 12 rotations of the Schulte-Wills realization that act as
    rigid motions of 3-space: the tetrahedral rotation group, as the
    cyclic coordinate permutations composed with the even sign
    changes.  Only these 12 of the 336 combinatorial symmetries of
    Klein's map survive as isometries of the embedding -- the rest
    act on the mesh but not on the metal, which is exactly why Egan's
    tetrahedral rendering emphasizes them."""
    mats = []
    for cyc in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        P = np.zeros((3, 3))
        for i, j in enumerate(cyc):
            P[i, j] = 1.0
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            mats.append(np.diag(np.asarray(signs, dtype=float)) @ P)
    return mats


def klein_quartic_tetra_classes():
    """Per-face class of the 56 triangles of the {3,7} view under the
    12 rigid rotations: 0 for the 8 CORNER triangles (an inward- and
    an outward-facing one at each of the lurking tetrahedron's 4
    corners -- the orbits of size 4, each face pinned by a 3-fold
    axis) and 1 for the 48 EDGE triangles (8 for each of the 6 edges
    -- the orbits of size 12).  This is the 56 = 8 + 48 decomposition
    Baez points out on Egan's tetrahedral picture.  Returns a list of
    56 zeros and ones; raises if the orbit structure comes out
    different, rather than colouring a lie."""
    V, F = build_klein_quartic(dual=False)
    va = np.asarray(V)
    vidx = {tuple(np.round(p, 6)): i for i, p in enumerate(va)}
    fid = {frozenset(f): i for i, f in enumerate(F)}
    fperms = []
    for M in klein_quartic_rotations():
        vperm = [vidx[tuple(np.round(M @ va[i], 6))]
                 for i in range(len(va))]
        fperms.append([fid[frozenset(vperm[a] for a in f)] for f in F])
    label = list(range(len(F)))

    def find(x):
        while label[x] != x:
            label[x] = label[label[x]]
            x = label[x]
        return x

    for fp in fperms:
        for i in range(len(F)):
            a, b = find(i), find(fp[i])
            if a != b:
                label[max(a, b)] = min(a, b)
    orbits = {}
    for i in range(len(F)):
        orbits.setdefault(find(i), []).append(i)
    sizes = sorted(len(o) for o in orbits.values())
    if sizes != [4, 4, 12, 12, 12, 12]:
        raise ValueError("tetrahedral orbit sizes %r" % sizes)
    classes = [1] * len(F)
    for o in orbits.values():
        if len(o) == 4:
            for i in o:
                classes[i] = 0
    return classes


def petrie_polygon_edges(F, start=0):
    """One Petrie polygon of the mesh as its cyclic edge list, traced
    flag by flag: the composed stride s2 s1 s0 is one zigzag step
    ('cross the edge, then turn'), and collecting each visited flag's
    edge until the stride returns to the start yields the closed
    left-right-left-right path.  On Klein's map the loop closes after
    8 edges -- Baez's devil's driving directions, LRLRLRLR and you
    are back where you began, and the 8 in the map's name {3,7}_8."""
    flags, s0, s1, s2 = _flag_system(F)
    x = start
    edges = []
    while True:
        edges.append(flags[x][1])
        x = s2[s1[s0[x]]]
        if x == start:
            break
        if len(edges) > 4 * len(flags):
            raise ValueError("Petrie walk failed to close")
    return edges


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
    # The cross-cap's dark seam.  It is non-orientable, so a closed
    # mesh of it MUST carry a ring of winding conflicts -- edges both
    # of whose faces traverse them the same way -- and averaging
    # normals across that ring renders it black.  The generator marks
    # the ring sharp; this pins the ring itself, since a change to the
    # RP^2 quotient that silently stopped identifying the equator
    # would remove the conflicts and the seam would come back with
    # nothing to mark.
    for nu_, nv_ in ((96, 48), (48, 24)):
        Vc, Fc = build_crosscap(nu_, nv_)
        ring = winding_conflict_edges(Fc)
        eq0_ = 1 + (nv_ - 2) * nu_
        assert len(ring) == nu_ // 2, (nu_, nv_, len(ring))
        assert all(a >= eq0_ and b >= eq0_ for a, b in ring), "ring off the equator"
    print("crosscap: winding-conflict ring is the identified equator, "
          "nu/2 edges OK")

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

    # ---- Dyck's surface: the torus-with-one-cross-cap form ----------
    # Von Dyck's theorem says T^2 # P^2 = 3 P^2, so BOTH forms must be
    # the same closed one-sided chi = -1 surface.  The sphere form is
    # already gated as N_3 in the loop above; the torus form is gated
    # here on the same three definitional facts, at several
    # resolutions (the surgery ring depends on the grid), plus a
    # flatness guard -- a mesh can pass every combinatorial check and
    # still be collapsed flat.
    bad = []
    for nseg, nring in ((64, 32), (48, 24), (40, 20), (96, 48)):
        V, F = build_dyck(nseg, nring)
        c = _chi(V, F)
        cnt = edge_face_counts(F)
        nb = sum(1 for x in cnt.values() if x == 1)
        if c != -1:
            bad.append('%dx%d:chi=%d' % (nseg, nring, c))
        if nb != 0:
            bad.append('%dx%d:boundary=%d' % (nseg, nring, nb))
        if _orientable(F):
            bad.append('%dx%d:two-sided' % (nseg, nring))
        if not np.all(np.isfinite(np.asarray(V))):
            bad.append('%dx%d:non-finite' % (nseg, nring))
    va = np.asarray(V)
    ext = va.max(axis=0) - va.min(axis=0)
    if float(ext.max()) > 4.0 * float(ext.min()):
        bad.append('collapsed: extents %s' % np.round(ext, 3))
    ok &= not bad
    print("topology: Dyck's surface (torus + one cross-cap) -- closed, "
          "one-sided, chi = -1 at 4 resolutions, not collapsed %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- the cap itself is a genuine cross-cap dome -----------------
    # A collapsed cap passes every combinatorial gate above: the
    # midpoint-weld version that once shipped had chi = -1, was
    # one-sided and closed, and rendered as a fan of slivers on a
    # spike -- every antipodal midpoint sits at the hole's centre, so
    # the cap had height but no width.  Gate on the geometry the
    # classical picture requires: the cap rises off the torus (max
    # radius R + r = 1.45 for the defaults), spreads in BOTH
    # transverse directions, and no face degenerates to a sliver.
    bad = []
    V, F = build_dyck(64, 32)
    va = np.asarray(V)
    hole = 0.60 * 0.45              # build_dyck's default cut radius
    amin = 1e9
    for f in F:
        p = va[list(f)]
        a = 0.0
        for i in range(1, len(f) - 1):
            a += 0.5 * float(np.linalg.norm(
                np.cross(p[i] - p[0], p[i + 1] - p[0])))
        amin = min(amin, a)
    capv = va[va[:, 0] > 1.45 + 1e-9]
    if amin < 1e-6:
        bad.append('sliver faces: min area %.2e' % amin)
    if len(capv) == 0:
        bad.append('no cap above the torus')
    else:
        rise = float(capv[:, 0].max()) - 1.45
        wy = float(np.ptp(capv[:, 1]))
        wz = float(np.ptp(capv[:, 2]))
        if rise < 0.25 * hole:
            bad.append('cap rise %.3f' % rise)
        if wy < 0.3 * hole or wz < 0.3 * hole:
            bad.append('cap collapsed: width %.3f x %.3f' % (wy, wz))
    ok &= not bad
    print("topology: Dyck cross-cap is a real dome -- min face area "
          "%.1e, rise %.2f, width %.2f x %.2f over hole %.2f %s"
          % (amin, float(capv[:, 0].max()) - 1.45 if len(capv) else 0,
             float(np.ptp(capv[:, 1])) if len(capv) else 0,
             float(np.ptp(capv[:, 2])) if len(capv) else 0, hole,
             'OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- the Klein quartic ------------------------------------------
    # What is claimed is not "a genus-3 mesh" but Klein's map itself,
    # so the gates are the map's own fingerprints: the {3,7} / {7,3}
    # counts, chi = -4, orientable and closed; the vertex construction
    # (two homothetic truncated tetrahedra, ratio 1/2); and above all
    # REGULARITY -- the flag-counting automorphism check must find all
    # 336 symmetries, and the Petrie length must be 8, the subscript
    # that names {3,7}_8.  A cube runs first as the control: 48
    # automorphisms and Petrie hexagons, or the counter proves nothing.
    import itertools
    bad = []
    Vp, Fp = build_klein_quartic(dual=False)
    Vd, Fd = build_klein_quartic(dual=True)
    for name, V, F, nv, nf, sides, val in (
            ('{3,7}', Vp, Fp, 24, 56, 3, 7),
            ('{7,3}', Vd, Fd, 56, 24, 7, 3)):
        cnt = edge_face_counts(F)
        if (len(V), len(cnt), len(F)) != (nv, 84, nf):
            bad.append('%s:counts %d/%d/%d'
                       % (name, len(V), len(cnt), len(F)))
        if _chi(V, F) != -4:
            bad.append('%s:chi=%d' % (name, _chi(V, F)))
        if any(x != 2 for x in cnt.values()):
            bad.append('%s:not closed' % name)
        if not _orientable(F):
            bad.append('%s:one-sided' % name)
        if winding_conflict_edges(F):
            bad.append('%s:winding' % name)
        if any(len(f) != sides for f in F):
            bad.append('%s:face sides' % name)
        deg = {}
        for f in F:
            for v in f:
                deg[v] = deg.get(v, 0) + 1
        if any(d != val for d in deg.values()):
            bad.append('%s:valence' % name)
        va = np.asarray(V)
        ext = va.max(axis=0) - va.min(axis=0)
        if float(ext.max()) > 1.1 * float(ext.min()):
            bad.append('%s:collapsed %s' % (name, np.round(ext, 3)))
    outer = {(sx * p[0], sy * p[1], sz * p[2])
             for p in set(itertools.permutations((1.0, 1.0, 3.0)))
             for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)
             if sx * sy * sz > 0}
    want = outer | {(x / 2, y / 2, z / 2) for (x, y, z) in outer}
    if set(Vp) != want:
        bad.append('vertices != two homothetic truncated tetrahedra')
    cube = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
            (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    if _map_automorphism_count(cube) != 48:
        bad.append('control: cube automorphisms != 48')
    if _map_petrie_lengths(cube) != {6}:
        bad.append('control: cube Petrie != 6')
    for name, F in (('{3,7}', Fp), ('{7,3}', Fd)):
        if _map_automorphism_count(F) != 336:
            bad.append('%s:automorphisms != 336' % name)
        if _map_petrie_lengths(F) != {8}:
            bad.append('%s:Petrie != 8' % name)
    ok &= not bad
    print("topology: Klein quartic -- {3,7} and {7,3} views closed, "
          "orientable, chi = -4, valences right, vertices = two "
          "homothetic truncated tetrahedra, and REGULAR: all 336 map "
          "automorphisms found flag-by-flag, Petrie length 8 (cube "
          "control 48 / 6) %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- flags, tetrahedral orbits, one Petrie polygon --------------
    # The claims are Baez's, checked as counts on the mesh: 24 x 14 =
    # 336 flag triangles (one per symmetry, reflections included),
    # splitting 168 + 168 by handedness in a strict checkerboard; the
    # 56 triangles fall 8 + 48 under the 12 rigid rotations ("2 for
    # each of the tetrahedron's 4 corners, and 8 for each of its 6
    # edges"), with every corner triangle pinned on a 3-fold axis;
    # and the traced Petrie polygon closes after exactly 8 distinct
    # chained edges -- the devil's driving directions.
    bad = []
    for name, dual in (('{3,7}', False), ('{7,3}', True)):
        V0, F0 = build_klein_quartic(dual=dual)
        V2, T2, info = subdivide_flags(V0, F0)
        if len(T2) != 336:
            bad.append('%s:flags=%d' % (name, len(T2)))
        if _chi(V2, T2) != -4:
            bad.append('%s:flag chi=%d' % (name, _chi(V2, T2)))
        if any(x != 2 for x in edge_face_counts(T2).values()):
            bad.append('%s:flag mesh not closed' % name)
        if len({(v, e, f) for v, e, f, _c in info}) != 336:
            bad.append('%s:duplicate flags' % name)
        ch = [c for _v, _e, _f, c in info]
        if (ch.count(0), ch.count(1)) != (168, 168):
            bad.append('%s:handedness %d/%d'
                       % (name, ch.count(0), ch.count(1)))
        ef2 = {}
        for ti, t in enumerate(T2):
            for i in range(3):
                a, b = t[i], t[(i + 1) % 3]
                ef2.setdefault((a, b) if a < b else (b, a),
                               []).append(ti)
        if any(ch[p[0]] == ch[p[1]] for p in ef2.values()):
            bad.append('%s:handedness not alternating' % name)
        pe = petrie_polygon_edges(F0)
        if len(pe) != 8 or len(set(pe)) != 8:
            bad.append('%s:petrie %d edges' % (name, len(pe)))
        elif not all(set(pe[i]) & set(pe[(i + 1) % 8])
                     for i in range(8)):
            bad.append('%s:petrie chain broken' % name)
    cls = klein_quartic_tetra_classes()
    if (cls.count(0), cls.count(1)) != (8, 48):
        bad.append('tetra split %d/%d' % (cls.count(0), cls.count(1)))
    Vp2, Fp2 = build_klein_quartic(dual=False)
    for fi, c in enumerate(cls):
        if c == 0:
            g = np.mean([Vp2[a] for a in Fp2[fi]], axis=0)
            g = g / np.linalg.norm(g)
            if float(np.max(np.abs(np.abs(g)
                                   - 1.0 / math.sqrt(3.0)))) > 1e-9:
                bad.append('corner face %d off axis' % fi)
    ok &= not bad
    print("topology: Klein quartic flags -- 336 per view, chi -4, "
          "handedness 168 + 168 checkerboarded; tetrahedral orbits "
          "8 corner + 48 edge triangles, corners on 3-fold axes; "
          "Petrie polygon = 8 chained edges %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("topology self-test failed")

