# Parametric surfaces from Ferreol's Encyclopedie des formes
# mathematiques remarquables.
#
# Part of the Math Art surfaces engine (`math_art/surfaces/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operators stay in their flat generator
# modules.
#
# These four briefly had an operator of their own, grouped because they
# came out of one book.  That is a bibliographic reason, not a
# mathematical one, and they are now served by the two operators whose
# subject each actually is: the Zoll surfaces by
# `curiosity_surface_generator` (Miscellaneous Surfaces), where the
# other classical named parametric surfaces live, and the Darboux
# surface by `helical_surface_generator` (Swept Surfaces), whose three
# existing formulas it generalises.  This module keeps the mathematics
# and the gates for both.
#
# Four surfaces (in two groups), each named for a property rather than a
# shape, and each gated on that property:
#
# ZOLL SURFACES -- every geodesic closes up.
#     On a round sphere every geodesic is a great circle: it comes back
#     to where it started.  That is a very fragile thing to ask of a
#     surface, and for a long time the sphere was the only known answer.
#     Tannery found another in 1892 and Zoll a third in 1903, and the
#     two of them are here.
#
#   Tannery's pear   x = a sin u cos u cos v
#                    y = a sin u cos u sin v
#                    z = 2 sqrt2 a sin u
#     the surface of revolution of half a Gerono lemniscate stretched by
#     2 sqrt2, equivalently the cylindrical 64 a^2 rho^2 = z^2(8a^2-z^2).
#     Its geodesics are closed algebraic curves shaped like bent figure
#     eights, all of length 2 pi a -- the same as a meridian -- except
#     the longest parallel, which is a geodesic of half that length.  It
#     is NOT smooth: both tips are conical points, and that is exactly
#     what Zoll's surface fixes.
#
#   Tannery's hourglass   the same formula run over u in [-pi/2, pi/2]
#     instead of [0, pi/2], so the whole figure eight is revolved rather
#     than half of it: two pears meeting tip to tip at the origin.
#
#   Zoll's surface   x = a sin u cos v, y = a sin u sin v,
#                    z = a integral_0^u sin U sqrt(1 + cos U
#                                    + sin^2 U cos^2 U / 4) dU
#     smooth, not the round sphere, and still every geodesic closes at
#     length 2 pi a.  The height integral is not elementary, so it is
#     accumulated numerically; the self-test confirms the result really
#     does carry the first fundamental form Zoll's construction asks
#     for, which is the thing that makes the geodesics close.
#
# DARBOUX SURFACES -- swept by a rigid curve.
#     A Darboux surface is a union of congruent copies of one curve:
#     kinematically, the trace of an unbending wire moved through space.
#     Translating the wire gives a translation surface, spinning it
#     about a fixed axis gives a surface of revolution, screwing it
#     gives a helicoid -- and a general rigid motion gives something
#     that is none of the three, which is the case the encyclopedia
#     singles out and the one the GENERAL mode builds: the wire spins
#     about the axis while tumbling, and its centre rides a circle.
#     The self-test measures the full matrix of pairwise distances
#     within each swept copy and confirms it never changes, which IS the
#     definition -- and which a non-orthogonal transport matrix (the way
#     this goes wrong) would immediately break.
#
# References:
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#   mathcurve.com, chapters "poire de Tannery", "surface de Darboux".
#   A converted copy of the encyclopedia is in research/books/
#   mathcurve_encyclopedie_formes_mathematiques/.
# - J. Tannery, Bulletin des sciences mathematiques, 2e serie, 16
#   (1892) 190 -- the pear.
# - O. Zoll, "Ueber Flaechen mit Scharen geschlossener geodaetischer
#   Linien", Mathematische Annalen 57 (1903) 108-133.
# - G. Darboux, "Lecons sur la theorie generale des surfaces", 1887-96
#   -- the surfaces swept by a rigid curve.
# - A. L. Besse, "Manifolds all of whose Geodesics are Closed",
#   Springer 1978 -- the modern account of Zoll surfaces.

import math

import numpy as np

_TWO_PI = 2.0 * math.pi
_SQRT8 = 2.0 * math.sqrt(2.0)


# ----------------------------------------------------------------------
# mesh plumbing
# ----------------------------------------------------------------------

def _grid_faces(nu, nv, wrap_u=False, wrap_v=False):
    """Quad faces over an (nu, nv) vertex grid, optionally cyclic."""
    faces = []
    iu = range(nu if wrap_u else nu - 1)
    iv = range(nv if wrap_v else nv - 1)
    for i in iu:
        i2 = (i + 1) % nu
        for j in iv:
            j2 = (j + 1) % nv
            faces.append((i * nv + j, i2 * nv + j,
                          i2 * nv + j2, i * nv + j2))
    return faces


def _weld_poles(P, faces, nu, nv):
    """Collapse rings that have degenerated to a single point.

    A surface of revolution closes at a pole by sending a whole ring of
    vertices to one place.  Left alone that ring is nv coincident
    vertices and a band of zero-area quads, which shades black and
    breaks any later solidify.  Only the FIRST and LAST rings can
    degenerate here, and each is checked rather than assumed, so a
    parametrisation that happens not to close is left untouched.
    """
    P = P.reshape(nu, nv, 3)
    keep = np.ones(nu * nv, dtype=bool)
    remap = np.arange(nu * nv)
    for ring in (0, nu - 1):
        r = P[ring]
        if float(np.max(np.abs(r - r[0]))) > 1e-9:
            continue
        base = ring * nv
        keep[base + 1:base + nv] = False
        remap[base + 1:base + nv] = base
    idx = np.cumsum(keep) - 1
    remap = idx[remap]
    out = []
    for f in faces:
        g = [int(remap[i]) for i in f]
        # drop repeated indices; a quad against a welded pole becomes a
        # triangle, and one that collapses entirely is dropped
        h = [g[0]] + [b for a, b in zip(g, g[1:]) if a != b]
        if len(h) > 2 and h[0] == h[-1]:
            h = h[:-1]
        if len(h) >= 3:
            out.append(tuple(h))
    return P.reshape(-1, 3)[keep], out


def _fit(V, scale=1.0):
    """Centre on the origin and fit the longest axis to 2 m."""
    V = np.asarray(V, dtype=float)
    if not len(V):
        return V
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    return (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-12 else 1.0) * scale


# ----------------------------------------------------------------------
# Zoll surfaces
# ----------------------------------------------------------------------

def tannery_point(u, v, a=1.0):
    """Tannery's pear/hourglass as a function of (u, v).

    rho = a sin u cos u is signed: for u < 0 it is negative, which under
    the revolution simply places the point on the opposite side, and is
    what turns the half figure eight into the whole one.
    """
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float),
                               np.asarray(v, dtype=float))
    rho = a * np.sin(u) * np.cos(u)
    return np.stack([rho * np.cos(v), rho * np.sin(v),
                     _SQRT8 * a * np.sin(u)], axis=-1)


#: Gauss-Legendre nodes and weights on [-1, 1], built once
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def zoll_height(u, a=1.0):
    """z(u) = a integral_0^u sin U sqrt(1 + cos U + sin^2U cos^2U/4) dU.

    Not elementary, so it is quadrature -- but Gauss-Legendre on
    [0, u] SEPARATELY FOR EACH u, not a cumulative sum interpolated
    afterwards.  That distinction matters here and cost a debugging
    pass: a linearly interpolated table of the integral has a piecewise
    CONSTANT derivative, so while z itself was accurate to 1e-7 the
    first fundamental form built from it was wrong in the fourth
    decimal, and it is the fundamental form that the closed geodesics
    depend on.  Integrating afresh per point removes the interpolant
    entirely.

    The integrand is analytic on [0, pi]: sin U vanishes at both ends,
    and although 1 + cos U + sin^2U cos^2U/4 has a double zero at U = pi
    its square root is then linear in (pi - U) rather than singular.  So
    a 48-node rule is exact to machine precision and no grading is
    needed.
    """
    u = np.asarray(u, dtype=float)
    half = 0.5 * u[..., None]
    U = half * (_GL_X + 1.0)                     # nodes on [0, u]
    c, s = np.cos(U), np.sin(U)
    f = s * np.sqrt(np.maximum(1.0 + c + s * s * c * c / 4.0, 0.0))
    return a * half[..., 0] * np.einsum('...k,k->...', f, _GL_W)


def zoll_point(u, v, a=1.0):
    """Zoll's smooth surface, all of whose geodesics close."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float),
                               np.asarray(v, dtype=float))
    return np.stack([a * np.sin(u) * np.cos(v),
                     a * np.sin(u) * np.sin(v),
                     zoll_height(u, a)], axis=-1)


def build_zoll(kind='TANNERY_PEAR', a=1.0, res_u=64, res_v=96, scale=1.0):
    """Mesh a Zoll surface.  Returns (verts, faces).

    kind: TANNERY_PEAR | TANNERY_HOURGLASS | ZOLL
    """
    if kind == 'TANNERY_PEAR':
        u0, u1, fn = 0.0, 0.5 * math.pi, tannery_point
    elif kind == 'TANNERY_HOURGLASS':
        u0, u1, fn = -0.5 * math.pi, 0.5 * math.pi, tannery_point
    elif kind == 'ZOLL':
        u0, u1, fn = 0.0, math.pi, zoll_point
    else:
        raise ValueError("unknown Zoll surface %r" % (kind,))
    nu, nv = max(4, int(res_u)) + 1, max(6, int(res_v))
    u = np.linspace(u0, u1, nu)
    v = np.linspace(0.0, _TWO_PI, nv, endpoint=False)
    P = fn(u[:, None], v[None, :], a).reshape(-1, 3)
    faces = _grid_faces(nu, nv, wrap_u=False, wrap_v=True)
    V, F = _weld_poles(P, faces, nu, nv)
    return _fit(V, scale), F


# ----------------------------------------------------------------------
# Darboux surfaces
# ----------------------------------------------------------------------
# A Darboux surface is (A(v) C(u) + T(v)) with A(v) orthogonal: one
# rigid curve C carried by a one-parameter motion.  The three classical
# specialisations and the general case differ only in what A and T are,
# so they share one builder and one congruence gate.

def _generatrix(kind, t, size=1.0, ratio=0.5):
    """The rigid curve, drawn in the xz-plane so that a rotation about
    Oz sweeps it into a surface rather than sliding it along itself."""
    if kind == 'CIRCLE':
        x, z = np.cos(t), np.sin(t)
    elif kind == 'ELLIPSE':
        x, z = np.cos(t), ratio * np.sin(t)
    elif kind == 'LEMNISCATE':                 # Gerono's figure eight
        x, z = np.cos(t), np.sin(t) * np.cos(t)
    elif kind == 'ASTROID':
        x, z = np.cos(t) ** 3, np.sin(t) ** 3
    elif kind == 'SEGMENT':                    # gives a ruled surface
        x, z = 2.0 * t / _TWO_PI - 1.0, np.zeros_like(t)
    else:
        raise ValueError("unknown generatrix %r" % (kind,))
    return size * np.stack([x, np.zeros_like(x), z], axis=-1)


def _motion(kind, v, radius=1.0, pitch=0.4, tilt=0.6, wobbles=3):
    """(A, T) for the motion at parameter v.

    A is built as a product of coordinate rotations, so it is orthogonal
    by construction rather than by normalisation -- which is what makes
    the congruence of the generatrices exact instead of approximate.
    """
    n = len(v)
    c, s = np.cos(v), np.sin(v)
    zero, one = np.zeros(n), np.ones(n)
    Rz = np.stack([np.stack([c, -s, zero], -1),
                   np.stack([s, c, zero], -1),
                   np.stack([zero, zero, one], -1)], -2)
    if kind == 'TRANSLATION':
        A = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        T = np.stack([zero, radius * v / math.pi, pitch * v], -1)
    elif kind == 'REVOLUTION':
        A, T = Rz, np.stack([zero, zero, zero], -1)
    elif kind == 'HELICOID':
        A, T = Rz, np.stack([zero, zero, pitch * v], -1)
    elif kind == 'GENERAL':
        # spin about Oz while tumbling about Ox, with the centre riding
        # a circle: none of translation, revolution or helicoid
        w = tilt * np.sin(max(1, int(wobbles)) * v)
        cw, sw = np.cos(w), np.sin(w)
        Rx = np.stack([np.stack([one, zero, zero], -1),
                       np.stack([zero, cw, -sw], -1),
                       np.stack([zero, sw, cw], -1)], -2)
        A = Rz @ Rx
        T = np.stack([radius * c, radius * s, zero], -1)
    elif kind == 'ROTOID':
        # The generalized helicoid: instead of screwing the curve about
        # a fixed AXIS, carry it along a spine CURVE in the spine's own
        # normal plane, turning it as it goes.  The spine here is a
        # circular helix of radius `radius` and pitch `pitch`, and
        # `tilt` is how many turns the curve makes per turn of the
        # spine.  Two knobs recover three classical surfaces:
        #   tilt = 0            a plain tube about the helix -- a COIL
        #   pitch = 0, tilt = 0 a tube about a circle -- a TORUS
        #   tilt != 0           the rotoid proper
        # A is assembled from the spine's own orthonormal Frenet frame,
        # so it is a rigid motion exactly and the swept curve stays
        # congruent to itself.
        T3 = np.stack([-radius * s, radius * c,
                       np.full(n, pitch)], -1)
        T3 = T3 / np.linalg.norm(T3, axis=1)[:, None]        # tangent
        N3 = np.stack([-c, -s, zero], -1)                    # to the axis
        B3 = np.cross(T3, N3)
        psi = tilt * v
        cp, sp = np.cos(psi), np.sin(psi)
        Np = cp[:, None] * N3 - sp[:, None] * B3
        Bp = sp[:, None] * N3 + cp[:, None] * B3
        # columns are the images of e1, e2, e3.  The generatrix is drawn
        # with y = 0, so only the first and third matter for the shape;
        # the middle one is taken as -T3 rather than +T3 so the frame is
        # right-handed and the motion is a rotation, not a reflection.
        A = np.stack([Np, -T3, Bp], axis=-1)
        T = np.stack([radius * c, radius * s, pitch * v], -1)
    else:
        raise ValueError("unknown motion %r" % (kind,))
    return A, T


#: the motions, and whether each is one of the three classical
#: specialisations (used by the self-test to check that GENERAL really
#: is none of them)
MOTIONS = (
    ('GENERAL', "General Darboux Motion"),
    ('TRANSLATION', "Translation"),
    ('REVOLUTION', "Revolution"),
    ('HELICOID', "Helicoidal"),
    ('ROTOID', "Rotoid (along a helix)"),
)

GENERATRICES = (
    ('CIRCLE', "Circle"),
    ('ELLIPSE', "Ellipse"),
    ('LEMNISCATE', "Gerono Lemniscate"),
    ('ASTROID', "Astroid"),
    ('SEGMENT', "Segment"),
)


def build_darboux(motion='GENERAL', generatrix='CIRCLE', size=0.45,
                  ratio=0.5, radius=1.0, pitch=0.4, tilt=0.6, wobbles=3,
                  turns=1.0, res_u=72, res_v=144, scale=1.0):
    """Mesh a Darboux surface: one rigid curve carried by a motion.

    Returns (verts, faces).  `turns` is how far the motion runs, in full
    revolutions; a whole number closes the sweep and the mesh is welded
    across the seam, anything else leaves it open.
    """
    nu = max(6, int(res_u))
    nv = max(6, int(res_v))
    # A whole number of turns only CLOSES the sweep if the motion
    # itself is periodic: revolution and the general (tumbling) motion
    # return to their start, but a screw or a helical spine climbs by
    # pitch per turn, so welding their seam would stitch faces across
    # the vertical gap.  (A helicoid or rotoid with zero pitch is a
    # revolution again and may close.)
    periodic = motion in ('REVOLUTION', 'GENERAL') or (
        motion in ('HELICOID', 'ROTOID') and abs(pitch) < 1e-12)
    closed_v = (periodic and abs(turns - round(turns)) < 1e-9
                and round(turns) >= 1)
    t = np.linspace(0.0, _TWO_PI, nu, endpoint=False)
    C = _generatrix(generatrix, t, size, ratio)          # (nu, 3)
    if generatrix == 'SEGMENT':
        # an open curve: sample it end to end rather than cyclically
        t = np.linspace(0.0, _TWO_PI, nu)
        C = _generatrix(generatrix, t, size, ratio)
    v = np.linspace(0.0, _TWO_PI * turns, nv,
                    endpoint=not closed_v)
    A, T = _motion(motion, v, radius, pitch, tilt, wobbles)
    # (nv, 3, 3) @ (3, nu) -> (nv, 3, nu); transpose to (nv, nu, 3)
    P = np.einsum('vij,uj->vui', A, C) + T[:, None, :]
    faces = _grid_faces(nv, nu, wrap_u=closed_v,
                        wrap_v=generatrix != 'SEGMENT')
    return _fit(P.reshape(-1, 3), scale), faces


# ----------------------------------------------------------------------
# the operator's table
# ----------------------------------------------------------------------
#: key -> (label, group, description).  Kept as the engine's own index
#: of what it holds -- the self-test walks it -- now that the two host
#: operators each carry only their own half in their enums.
PRESETS = {
    'TANNERY_PEAR': (
        "Tannery's Pear", 'ZOLL',
        "The surface of revolution of half a Gerono lemniscate "
        "stretched by 2 sqrt2. Every geodesic closes, and all but one "
        "have the same length 2 pi a as a meridian; both tips are "
        "conical points"),
    'TANNERY_HOURGLASS': (
        "Tannery's Hourglass", 'ZOLL',
        "The whole Gerono lemniscate revolved instead of half of it: "
        "two pears meeting tip to tip"),
    'ZOLL': (
        "Zoll's Surface", 'ZOLL',
        "Zoll's 1903 answer to Tannery: smooth, not a round sphere, "
        "and still every geodesic closes at length 2 pi a"),
    'DARBOUX': (
        "Darboux Surface", 'DARBOUX',
        "A rigid curve swept by a motion. Translation, revolution and "
        "screw motions give translation surfaces, surfaces of "
        "revolution and helicoids; the general motion gives a Darboux "
        "surface that is none of the three"),
}

PRESET_ORDER = ('TANNERY_PEAR', 'TANNERY_HOURGLASS', 'ZOLL', 'DARBOUX')


def build_preset(key, res_u=64, res_v=96, a=1.0, scale=1.0, **kw):
    """Mesh a named surface with the operator's parameters."""
    if key == 'DARBOUX':
        return build_darboux(res_u=res_u, res_v=res_v, scale=scale, **kw)
    return build_zoll(key, a=a, res_u=res_u, res_v=res_v, scale=scale)


# ----------------------------------------------------------------------

def _selftest():
    """Each surface is gated on the property it is named for.

    1. Tannery's pear carries the first fundamental form Tannery
       printed, a^2((2+cos2u)^2 du^2 + sin^2u cos^2u dtheta^2), and
       satisfies its own cylindrical equation.  The metric is what makes
       the geodesics close, so it is the right thing to check -- and it
       is sensitive to the 2 sqrt2 stretch, which is the one constant
       here that a transcription would get wrong.
    2. Its geodesics really do close at length 2 pi a: integrated
       directly from the geodesic equations for a surface of
       revolution, using Clairaut's relation.
    3. Zoll's height integral reproduces the first fundamental form
       Zoll's construction asks for.
    4. Every Darboux generatrix is congruent to every other, measured
       as the full matrix of pairwise distances within each swept copy
       -- which IS the definition of a Darboux surface.
    5. The GENERAL motion is genuinely none of the three classical
       specialisations.
    6. Every preset meshes to a solid, finite, watertight-where-it-
       should-be patch in the 2 m cube.
    """
    ok = True

    # ---- 1. Tannery's metric ----------------------------------------
    a = 1.3
    rng = np.random.default_rng(20260821)
    uu = rng.uniform(0.05, 0.5 * math.pi - 0.05, 200)
    vv = rng.uniform(0.0, _TWO_PI, 200)
    h = 1e-6

    def deriv(fn, u, v, du, dv):
        return (fn(u + du * h, v + dv * h, a)
                - fn(u - du * h, v - dv * h, a)) / (2.0 * h)

    Pu = deriv(tannery_point, uu, vv, 1.0, 0.0)
    Pv = deriv(tannery_point, uu, vv, 0.0, 1.0)
    E = np.einsum('ij,ij->i', Pu, Pu)
    F = np.einsum('ij,ij->i', Pu, Pv)
    G = np.einsum('ij,ij->i', Pv, Pv)
    wantE = a * a * (2.0 + np.cos(2.0 * uu)) ** 2
    wantG = a * a * np.sin(uu) ** 2 * np.cos(uu) ** 2
    P = tannery_point(uu, vv, a)
    rho2 = P[:, 0] ** 2 + P[:, 1] ** 2
    z = P[:, 2]
    checks = [
        ("E = a^2(2+cos2u)^2",
         float(np.max(np.abs(E - wantE))) / float(np.max(wantE))),
        ("G = a^2 sin^2u cos^2u",
         float(np.max(np.abs(G - wantG))) / float(np.max(wantG))),
        ("F = 0", float(np.max(np.abs(F))) / float(np.max(wantE))),
        ("64 a^2 rho^2 = z^2 (8a^2 - z^2)",
         float(np.max(np.abs(64.0 * a * a * rho2
                             - z * z * (8.0 * a * a - z * z))))
         / (64.0 * a ** 4)),
    ]
    bad = ['%s:%.1e' % (n, d) for n, d in checks if not (d < 1e-8)]
    ok &= not bad
    print("encyclopedia: Tannery's pear carries Tannery's metric %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- 2. the geodesics close at 2 pi a ---------------------------
    # For a surface of revolution with metric E(u) du^2 + G(u) dtheta^2,
    # Clairaut gives G dtheta/ds = const = c, and E (du/ds)^2 = 1 - c^2/G.
    # Integrating arc length between the two turning points (where
    # G = c^2) and doubling gives the length of the closed geodesic.
    def tannery_geodesic_length(c, n=200001):
        # G = a^2 sin^2u cos^2u = a^2 sin^2(2u)/4, so the two turning
        # points where G = c^2 are available in CLOSED FORM.  Taking
        # them off a grid instead leaves the interval short by up to one
        # step at each end, and since the integrand blows up like
        # 1/sqrt there, that costs sqrt(h) -- which is what a first
        # attempt here measured as a 1.2e-3 "error" in a length that is
        # in fact exact.
        s = 2.0 * c / a
        if not (0.0 < s < 1.0):
            return None
        u0 = 0.5 * math.asin(s)
        u1 = 0.5 * math.pi - u0
        # ds = sqrt(E/(1 - c^2/G)) du has an inverse-square-root
        # singularity at each end; the cosine substitution
        # u = u0 + (u1-u0)(1-cos th)/2 turns both into a finite
        # integrand, since du/dth vanishes there at the matching rate.
        th = np.linspace(0.0, math.pi, n)
        u = u0 + 0.5 * (u1 - u0) * (1.0 - np.cos(th))
        Ef = a * a * (2.0 + np.cos(2.0 * u)) ** 2
        Gf = a * a * np.sin(u) ** 2 * np.cos(u) ** 2
        w = np.sqrt(np.maximum(1.0 - c * c / Gf, 0.0))
        du_dth = 0.5 * (u1 - u0) * np.sin(th)
        f = np.where(w > 1e-300, np.sqrt(Ef) * du_dth / np.maximum(w, 1e-300),
                     0.0)
        # the endpoints are 0/0; fill them by the limit from within
        f[0], f[-1] = f[1], f[-2]
        return 2.0 * float(np.trapezoid(f, th))

    # the maximum of G is at u = pi/4, G = a^2/4, so c < a/2
    lens = [tannery_geodesic_length(c) for c in (0.05 * a, 0.15 * a,
                                                 0.3 * a, 0.45 * a)]
    want = _TWO_PI * a
    devs = [abs(L - want) / want for L in lens if L is not None]
    good = len(devs) == 4 and max(devs) < 1e-5
    ok &= good
    print("encyclopedia: Tannery geodesics all close at 2 pi a "
          "(worst dev %.1e over 4 Clairaut constants) %s"
          % (max(devs) if devs else float('nan'), 'OK' if good else 'FAIL'))

    # ---- 3. Zoll's metric -------------------------------------------
    uu = rng.uniform(0.02, math.pi - 0.02, 300)
    vv = rng.uniform(0.0, _TWO_PI, 300)
    Pu = deriv(zoll_point, uu, vv, 1.0, 0.0)
    Pv = deriv(zoll_point, uu, vv, 0.0, 1.0)
    E = np.einsum('ij,ij->i', Pu, Pu)
    F = np.einsum('ij,ij->i', Pu, Pv)
    G = np.einsum('ij,ij->i', Pv, Pv)
    wantE = a * a * (1.0 + np.cos(uu) * np.sin(uu) ** 2 / 2.0) ** 2
    wantG = a * a * np.sin(uu) ** 2
    checks = [
        ("E = a^2(1 + cos u sin^2u / 2)^2",
         float(np.max(np.abs(E - wantE))) / float(np.max(wantE))),
        ("G = a^2 sin^2 u",
         float(np.max(np.abs(G - wantG))) / float(np.max(wantG))),
        ("F = 0", float(np.max(np.abs(F))) / float(np.max(wantE))),
    ]
    bad = ['%s:%.1e' % (n, d) for n, d in checks if not (d < 1e-5)]
    ok &= not bad
    print("encyclopedia: Zoll's height integral gives Zoll's metric %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- 4. the Darboux generatrices are congruent ------------------
    # The definition, measured directly: sample the swept copy at each
    # v and compare its full pairwise-distance matrix with the base
    # curve's.  Nothing about the motion is assumed.
    bad = []
    for mot, _lab in MOTIONS:
        for gen, _g in GENERATRICES:
            t = np.linspace(0.0, _TWO_PI, 24, endpoint=False)
            C = _generatrix(gen, t, 0.45, 0.5)
            D0 = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
            v = np.linspace(0.0, _TWO_PI, 17)
            A, T = _motion(mot, v, 1.0, 0.4, 0.6, 3)
            P = np.einsum('vij,uj->vui', A, C) + T[:, None, :]
            D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
            d = float(np.max(np.abs(D - D0[None]))) / float(np.max(D0))
            if not (d < 1e-12):
                bad.append('%s/%s:%.1e' % (mot, gen, d))
    ok &= not bad
    print("encyclopedia: %d Darboux motion/generatrix pairs sweep "
          "congruent copies %s"
          % (len(MOTIONS) * len(GENERATRICES),
             'OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- 5. GENERAL is none of the three ----------------------------
    v = np.linspace(0.0, _TWO_PI, 33)
    Ag, Tg = _motion('GENERAL', v, 1.0, 0.4, 0.6, 3)
    apart = []
    for mot, _lab in MOTIONS[1:]:
        Am, Tm = _motion(mot, v, 1.0, 0.4, 0.6, 3)
        apart.append((mot, float(np.max(np.abs(Ag - Am)))
                      + float(np.max(np.abs(Tg - Tm)))))
    bad = ['%s:%.1e' % (n, d) for n, d in apart if not (d > 1e-3)]
    # and the rotation part really is orthogonal
    orth = float(np.max(np.abs(
        np.einsum('vij,vkj->vik', Ag, Ag) - np.eye(3)[None])))
    det = float(np.min(np.linalg.det(Ag)))
    good = not bad and orth < 1e-12 and det > 0.999
    ok &= good
    print("encyclopedia: the general Darboux motion is none of the "
          "three classical ones, and stays a rotation "
          "(orthogonality %.1e, det %.6f) %s"
          % (orth, det, 'OK' if good else 'FAIL ' + ','.join(bad)))

    # ---- 6. everything meshes ---------------------------------------
    bad = []
    for key in PRESET_ORDER:
        V, F = build_preset(key, res_u=40, res_v=64)
        if len(V) < 50 or len(F) < 50 or not np.all(np.isfinite(V)):
            bad.append('%s:%d/%d' % (key, len(V), len(F)))
            continue
        ext = V.max(axis=0) - V.min(axis=0)
        if float(ext.max()) > 2.0 + 1e-6:
            bad.append(key + '(oversize)')
        elif float(ext.min() / ext.max()) < 0.02:
            bad.append('%s(thin %.3f)' % (key, ext.min() / ext.max()))
        # no degenerate faces left behind by the pole weld
        elif any(len(set(f)) != len(f) for f in F):
            bad.append(key + '(repeated index in a face)')
    ok &= not bad
    print("encyclopedia: %d presets mesh solid and finite in the 2 m "
          "cube %s" % (len(PRESET_ORDER),
                       'OK' if not bad else 'FAIL ' + ','.join(bad)))

    # the pear and the hourglass must actually close up: a surface of
    # revolution welded at its poles has no boundary edge at all.
    bad = []
    for key in ('TANNERY_PEAR', 'TANNERY_HOURGLASS', 'ZOLL'):
        V, F = build_preset(key, res_u=32, res_v=48)
        edges = {}
        for f in F:
            for i in range(len(f)):
                e = (f[i], f[(i + 1) % len(f)])
                edges[tuple(sorted(e))] = edges.get(tuple(sorted(e)), 0) + 1
        nb = sum(1 for c in edges.values() if c != 2)
        if nb:
            bad.append('%s:%d' % (key, nb))
    ok &= not bad
    print("encyclopedia: the three Zoll surfaces close up watertight %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("encyclopedia surfaces self-test failed")
