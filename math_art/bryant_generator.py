
# Bryant Surfaces Generator for Blender -- CMC-1 surfaces in hyperbolic
# space H^3.
#
# Everything else in this add-on lives in Euclidean R^3.  These surfaces
# do not: they are surfaces of constant mean curvature ONE in hyperbolic
# 3-space, drawn here through a model (Poincare ball by default) that
# packs all of H^3 into a finite picture.  H = 1 is the distinguished
# value in H^3 the way H = 0 is in R^3 -- it is the borderline at which
# a CMC surface stops behaving like a soap bubble and starts behaving
# like a minimal surface, and Bryant's theorem says these "Bryant
# surfaces" have a Weierstrass-type representation exactly as minimal
# surfaces in R^3 do.
#
# THE REPRESENTATION (Bryant 1987, Theorem A).  Model H^3 as the
# positive-definite Hermitian 2x2 matrices of determinant 1,
#
#     f = [[x0 + x3, x1 + i x2], [x1 - i x2, x0 - x3]] ,
#     det f = x0^2 - x1^2 - x2^2 - x3^2 = 1 ,
#
# which is the upper sheet of the Minkowski hyperboloid <x,x> = -1 with
# <a,b> = -a0 b0 + a1 b1 + a2 b2 + a3 b3.  Then for ANY holomorphic null
# immersion F into Sl(2,C) -- null meaning det(F^{-1} dF) = 0 --
#
#     f = F (conj F)^T
#
# is a conformal immersion into H^3 of mean curvature 1, and every such
# surface arises this way.  The nullity condition is the exact analogue
# of the Weierstrass data being null in R^3, and it is why these surfaces
# are as computable as minimal surfaces.
#
# FOUR FAMILIES ARE BUILT HERE:
#
#   CATENOID_COUSIN  Bryant's Example 2, a surface of REVOLUTION with a
#                    real parameter mu > -1/2, mu != 0.  Its profile is
#                    embedded for -1/2 < mu < 0 and has exactly one
#                    self-intersection for mu > 0 -- a visible, checkable
#                    change of shape at mu = 0.  Total curvature
#                    -4 pi (2 mu + 1), which unlike the Euclidean case is
#                    NOT quantised.
#   ENNEPER_COUSIN   Bryant's Example 1.  Its induced metric is the one
#                    Enneper's surface induces on C, hence the name, and
#                    its total curvature is -4 pi.
#   POLYNOMIAL       Bryant's Theorem B: for polynomials r1, r2 with no
#                    common zero, integrating
#                        F' = F [[r1 r2, -r2^2], [r1^2, -r1 r2]]
#                    from F(0) = I gives a complete CMC-1 immersion of C
#                    with total curvature -4 pi k.  Exposed here as
#                    r1 = z^n, r2 = c, which contains Enneper's cousin at
#                    n = 1 and generalises it upward.
#   TRINOID          Bobenko-Pavlyukevich-Springborn: the complete CMC-1
#                    surfaces of genus 0 with THREE catenoidal ends, in
#                    closed form via Gauss hypergeometric functions.  A
#                    three-parameter family in the moduli (d0, d1, d_inf)
#                    of the ends, admissible exactly when the monodromy
#                    of the underlying Fuchsian system is unitarizable
#                    (their Theorem 6 / Proposition 2); inadmissible
#                    parameters are refused, not approximated.  The
#                    numeric core lives in math_art/trinoid.py; the
#                    symmetric family d0 = d1 = d_inf changes from
#                    embedded to self-intersecting at d0 ~ 0.2332.
#
# Example 2 is closed form.  Writing z = r e^{i theta} and rho = |z|^2,
# the matrix product f = F (conj F)^T collapses to
#
#     (2mu+1) f00 = (mu+1)^2 r^{2mu} + mu^2 r^{-2mu-2}
#     (2mu+1) f11 = mu^2 r^{2mu+2} + (mu+1)^2 r^{-2mu}
#     (2mu+1) f01 = mu(mu+1) e^{-i theta} (r^{2mu+1} + r^{-2mu-1})
#
# and det f = 1 holds IDENTICALLY, because
# (mu+1)^4 + mu^4 - 2 mu^2 (mu+1)^2 = ((mu+1)^2 - mu^2)^2 = (2mu+1)^2.
# So the immersion lands exactly on the hyperboloid with no
# renormalisation, which is worth knowing: any drift from det f = 1 in a
# render is a bug, not a tolerance.
#
# MODELS.  A point of H^3 is drawn through one of
#
#     Poincare ball    (x1,x2,x3)/(1 + x0)   conformal, unit ball
#     Klein ball       (x1,x2,x3)/x0         geodesics are straight
#     Hyperboloid      (x1,x2,x3)            faithful but unbounded
#
# The ball models are the useful ones: a COMPLETE surface in H^3 has
# infinite Euclidean extent, so on the hyperboloid it runs off to
# infinity and the mesh is mostly wasted on the ends.  In a ball model
# the ends converge to the sphere at infinity and the whole surface fits
# in the picture -- which is also what makes the ends legible, since a
# CMC-1 end approaches a round circle on S^2_infinity.
#
# References:
# - Robert L. Bryant, "Surfaces of mean curvature one in hyperbolic
#   space", Asterisque 154-155 (1987), 321-347.  Theorem A (the
#   representation f = F (conj F)^T), Theorem B (the polynomial data on
#   C), Example 1 (Enneper's cousin, p. 341), Example 2 (the catenoid
#   cousins, pp. 341-342).
# - A. I. Bobenko, T. V. Pavlyukevich, B. A. Springborn, "Hyperbolic
#   constant mean curvature one surfaces: spinor representation and
#   trinoids in hypergeometric functions", Math. Z. 245 (2003), 63-91;
#   arXiv:math/0206021 -- the spinor form of the same representation and
#   the explicit trinoids the TRINOID mode implements (see
#   math_art/trinoid.py for the construction, gates and details).
# - M. Umehara and K. Yamada, "Complete surfaces of constant mean
#   curvature 1 in the hyperbolic 3-space", Ann. of Math. 137 (1993),
#   611-638 -- the global theory these examples opened up.

bl_info = {
    "name": "Bryant Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Bryant Surface",
    "description": "Constant-mean-curvature-one surfaces in hyperbolic "
                   "3-space, drawn in the Poincare or Klein ball",
    "category": "Add Mesh",
}

import math

import numpy as np

from . import trinoid as _trinoid

MODES = ('CATENOID_COUSIN', 'ENNEPER_COUSIN', 'POLYNOMIAL', 'TRINOID')
MODELS = ('POINCARE', 'KLEIN', 'HYPERBOLOID')

# Minkowski metric of R^{3,1}, signature (-,+,+,+)
_G = np.array([-1.0, 1.0, 1.0, 1.0])


def minkowski(a, b):
    """<a, b> = -a0 b0 + a1 b1 + a2 b2 + a3 b3, over trailing axis."""
    return np.sum(a * b * _G, axis=-1)


def hyperboloid_residual(X):
    """RELATIVE deviation from <x,x> = -1.

    Relative, not absolute, and the distinction is not pedantry: a
    complete surface in H^3 runs off towards the sphere at infinity, so
    its Minkowski coordinates grow without bound while <x,x> stays -1.
    The residual is then a difference of large nearly-equal numbers, and
    an absolute tolerance silently becomes a limit on how far out the
    surface may reach.  Dividing by the size of the cancelling terms
    measures the arithmetic instead of the geometry."""
    scale = np.sum(X * X, axis=-1)          # Euclidean, i.e. |terms|
    return np.abs(minkowski(X, X) + 1.0) / np.maximum(scale, 1.0)


def hermitian_to_hyperboloid(f00, f11, f01):
    """Hermitian (f00, f11, f01) with det = 1 -> Minkowski (x0,x1,x2,x3)
    on the upper sheet of <x,x> = -1."""
    x0 = 0.5 * (f00 + f11)
    x3 = 0.5 * (f00 - f11)
    return np.stack([x0, np.real(f01), np.imag(f01), x3], axis=-1)


def to_model(X, model='POINCARE'):
    """Project hyperboloid points to a drawable model."""
    if model == 'POINCARE':
        return X[..., 1:] / (1.0 + X[..., :1])
    if model == 'KLEIN':
        return X[..., 1:] / np.maximum(X[..., :1], 1e-12)
    if model == 'HYPERBOLOID':
        return X[..., 1:].copy()
    raise ValueError(f"unknown model {model!r}")


# --------------------------------------------------------------------------
# the three families, each as (u, v) -> hyperboloid point
# --------------------------------------------------------------------------

def catenoid_cousin(S, TH, mu=-0.3):
    """Bryant Example 2 on z = e^S e^{i TH}.  Closed form; see header."""
    m = float(mu)
    d = 2.0 * m + 1.0
    r = np.exp(S)
    f00 = ((m + 1.0) ** 2 * r ** (2 * m) + m * m * r ** (-2 * m - 2)) / d
    f11 = (m * m * r ** (2 * m + 2) + (m + 1.0) ** 2 * r ** (-2 * m)) / d
    amp = m * (m + 1.0) * (r ** (2 * m + 1) + r ** (-2 * m - 1)) / d
    f01 = amp * np.exp(-1j * TH)
    return hermitian_to_hyperboloid(f00, f11, f01)


def _f_from_F(F):
    """f = F (conj F)^T for a stack of 2x2 complex matrices, returned as
    (f00, f11, f01)."""
    a, b, c, d = F[..., 0, 0], F[..., 0, 1], F[..., 1, 0], F[..., 1, 1]
    f00 = a * np.conj(a) + b * np.conj(b)
    f11 = c * np.conj(c) + d * np.conj(d)
    f01 = a * np.conj(c) + b * np.conj(d)
    return np.real(f00), np.real(f11), f01


def enneper_cousin(X, Y, lam=1.0):
    """Bryant Example 1 on z = X + iY, with lambda real positive."""
    z = X + 1j * Y
    lz = lam * z
    ch, sh = np.cosh(lz), np.sinh(lz)
    F = np.empty(z.shape + (2, 2), dtype=complex)
    F[..., 0, 0] = ch - lz * sh
    F[..., 0, 1] = lam * sh
    F[..., 1, 0] = sh / lam - z * ch
    F[..., 1, 1] = ch
    return hermitian_to_hyperboloid(*_f_from_F(F))


def _poly_A(z, degree, c):
    """The Theorem B potential [[r1 r2, -r2^2], [r1^2, -r1 r2]] for
    r1 = z^degree, r2 = c."""
    r1 = z ** degree
    r2 = np.full_like(z, complex(c))
    A = np.empty(z.shape + (2, 2), dtype=complex)
    A[..., 0, 0] = r1 * r2
    A[..., 0, 1] = -r2 * r2
    A[..., 1, 0] = r1 * r1
    A[..., 1, 1] = -r1 * r2
    return A


def polynomial_bryant(X, Y, degree=1, c=1.0, steps=64):
    """Bryant Theorem B with r1 = z^degree, r2 = c.

    F' = F A(z) is integrated with RK4 from F(0) = I along the RAY to
    each sample point.  The equation is holomorphic and C is simply
    connected, so the value is path independent and the radial path is
    the cheapest one that reaches every point.  `steps` is per ray."""
    z = X + 1j * Y
    shape = z.shape
    F = np.zeros(shape + (2, 2), dtype=complex)
    F[..., 0, 0] = 1.0
    F[..., 1, 1] = 1.0
    h = (z / steps)[..., None, None]

    def deriv(Fm, zz):
        return Fm @ _poly_A(zz, degree, c)

    for i in range(steps):
        t0 = (i / steps)
        z0 = z * t0
        k1 = deriv(F, z0)
        k2 = deriv(F + 0.5 * h * k1, z * (t0 + 0.5 / steps))
        k3 = deriv(F + 0.5 * h * k2, z * (t0 + 0.5 / steps))
        k4 = deriv(F + h * k3, z * (t0 + 1.0 / steps))
        F = F + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return hermitian_to_hyperboloid(*_f_from_F(F))


def surface_points(mode, u, v, mu=-0.3, lam=1.0, degree=1, c=1.0,
                   steps=64, d0=0.16, d1=0.18, dinf=0.20, domain=1):
    """Hyperboloid points for a parameter grid (u, v)."""
    if mode == 'CATENOID_COUSIN':
        return catenoid_cousin(u, v, mu)
    if mode == 'ENNEPER_COUSIN':
        return enneper_cousin(u, v, lam)
    if mode == 'POLYNOMIAL':
        return polynomial_bryant(u, v, degree, c, steps)
    if mode == 'TRINOID':
        # (u, v) is the log-polar chart of one of the three end domains
        return _trinoid.trinoid_points(u, v, d0, d1, dinf, domain)
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------
# the acceptance test: mean curvature measured IN H^3
# --------------------------------------------------------------------------

def mean_curvature_h3(mode, u, v, h=1e-4, **kw):
    """Mean curvature of the immersed surface, computed in the Minkowski
    hyperboloid model -- so it measures curvature in H^3, not in the
    drawing model.

    The surface normal is the vector n with <n, X> = <n, X_u> =
    <n, X_v> = 0 and <n, n> = 1; since the ambient connection of H^3 is
    the Minkowski one projected onto X^perp, and n is already orthogonal
    to X, the second fundamental form is just <X_uu, n> and friends.
    Then H = (E N - 2 F M + G L) / (2 (E G - F^2)).

    The sign of n is arbitrary, so the invariant is |H|; Bryant's
    surfaces must give |H| = 1."""
    def P(a, b):
        return surface_points(mode, a, b, **kw)

    Xu = (P(u + h, v) - P(u - h, v)) / (2 * h)
    Xv = (P(u, v + h) - P(u, v - h)) / (2 * h)
    X0 = P(u, v)
    Xuu = (P(u + h, v) - 2 * X0 + P(u - h, v)) / (h * h)
    Xvv = (P(u, v + h) - 2 * X0 + P(u, v - h)) / (h * h)
    Xuv = (P(u + h, v + h) - P(u + h, v - h)
           - P(u - h, v + h) + P(u - h, v - h)) / (4 * h * h)

    # n = G w with w the Euclidean null vector of [X; Xu; Xv]
    stack = np.stack([X0, Xu, Xv], axis=-2)          # (..., 3, 4)
    flat = stack.reshape(-1, 3, 4)
    n = np.empty((flat.shape[0], 4))
    for i in range(flat.shape[0]):
        w = np.linalg.svd(flat[i])[2][-1]
        n[i] = w * _G
    n = n.reshape(X0.shape)
    nn = minkowski(n, n)
    n = n / np.sqrt(np.abs(nn))[..., None]

    E = minkowski(Xu, Xu)
    F = minkowski(Xu, Xv)
    G = minkowski(Xv, Xv)
    L = minkowski(Xuu, n)
    M = minkowski(Xuv, n)
    N = minkowski(Xvv, n)
    return (E * N - 2.0 * F * M + G * L) / (2.0 * (E * G - F * F))


# --------------------------------------------------------------------------
# meshing
# --------------------------------------------------------------------------

DOMAINS = {
    # (u range, v range, wrap in v)
    'CATENOID_COUSIN': ((-2.2, 2.2), (0.0, 2.0 * math.pi), True),
    'ENNEPER_COUSIN': ((-1.6, 1.6), (-1.6, 1.6), False),
    'POLYNOMIAL': ((-1.5, 1.5), (-1.5, 1.5), False),
    # log-polar chart of one end domain, u = log|wt| (kept off the
    # domain-boundary circle |wt| = 1, where the corner umbilics live)
    'TRINOID': ((math.log(0.04), -0.05), (0.0, 2.0 * math.pi), True),
}


def build_surface(mode='CATENOID_COUSIN', ures=96, vres=96,
                  model='POINCARE', mu=-0.3, lam=1.0, degree=1, c=1.0,
                  extent=1.0, steps=64, scale=1.0,
                  d0=0.16, d1=0.18, dinf=0.20, rmin=0.03):
    """Mesh a Bryant surface in the chosen model of H^3."""
    if mode == 'TRINOID':
        # three welded end-domain annuli; extent scales the end depth
        rm = float(np.clip(rmin ** extent, 1e-3, 0.5))
        P, faces, info = _trinoid.build_trinoid_mesh(
            d0, d1, dinf, nr=ures, nth=vres, rmin=rm, model=model)
        det_dev = info['det_dev']
    else:
        (u0, u1), (v0, v1), wrap = DOMAINS[mode]
        u0, u1 = u0 * extent, u1 * extent
        if not wrap:
            v0, v1 = v0 * extent, v1 * extent
        us = np.linspace(u0, u1, ures)
        vs = (np.linspace(v0, v1, vres, endpoint=False) if wrap
              else np.linspace(v0, v1, vres))
        U, V = np.meshgrid(us, vs, indexing='ij')
        X = surface_points(mode, U, V, mu=mu, lam=lam, degree=degree, c=c,
                           steps=steps)
        P = to_model(X, model).reshape(-1, 3)
        faces = []
        for i in range(ures - 1):
            for j in range(vres - (0 if wrap else 1)):
                j1 = (j + 1) % vres
                faces.append((i * vres + j, i * vres + j1,
                              (i + 1) * vres + j1, (i + 1) * vres + j))
        det_dev = float(hyperboloid_residual(X).max())
    lo, hi = P.min(axis=0), P.max(axis=0)
    ext = float((hi - lo).max())
    P = (P - 0.5 * (lo + hi)) * ((2.0 / ext if ext > 1e-9 else 1.0) * scale)
    return P, faces, {'det_dev': det_dev}


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

    class MESH_OT_bryant_surface_add(bpy.types.Operator):
        """Add a Bryant surface: constant mean curvature ONE in
        hyperbolic 3-space, drawn in the Poincare or Klein ball"""
        bl_idname = "mesh.bryant_surface_add"
        bl_label = "Bryant Surface"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Surface",
            description="Which Bryant CMC-1 surface family to build",
            items=[('CATENOID_COUSIN', "Catenoid Cousin",
                    "Bryant's Example 2: a CMC-1 surface of revolution, "
                    "embedded for -1/2 < mu < 0 and once "
                    "self-intersecting for mu > 0"),
                   ('ENNEPER_COUSIN', "Enneper Cousin",
                    "Bryant's Example 1: the CMC-1 surface whose induced "
                    "metric is Enneper's, of total curvature -4 pi"),
                   ('POLYNOMIAL', "Polynomial Data",
                    "Bryant's Theorem B: integrate F' = F A(z) for "
                    "r1 = z^n, r2 = c -- Enneper's cousin at n = 1 and "
                    "higher-order cousins above it"),
                   ('TRINOID', "Trinoid",
                    "Bobenko-Pavlyukevich-Springborn: the CMC-1 "
                    "surfaces with three catenoidal ends, in closed "
                    "form via hypergeometric functions.  The end "
                    "moduli (d0, d1, d_inf) must satisfy the "
                    "unitarizable-monodromy condition; symmetric "
                    "trinoids (all equal) are embedded below "
                    "d0 = 0.2332 and self-intersect above it")],
            default='CATENOID_COUSIN')
        model: EnumProperty(
            name="Model",
            description="Model of hyperbolic space the surface is drawn "
                        "in",
            items=[('POINCARE', "Poincare Ball",
                    "conformal unit-ball model: angles are true and the "
                    "whole surface fits in the picture"),
                   ('KLEIN', "Klein Ball",
                    "projective unit-ball model: geodesics are straight, "
                    "angles are not true"),
                   ('HYPERBOLOID', "Hyperboloid",
                    "the Minkowski hyperboloid's space part, undistorted "
                    "but unbounded")],
            default='POINCARE')
        mu: FloatProperty(
            name="mu", default=-0.3, min=-0.49, max=3.0,
            description="Catenoid-cousin parameter, mu > -1/2 and "
                        "mu != 0.  Total curvature is -4 pi (2 mu + 1); "
                        "the profile is embedded below 0 and has one "
                        "self-intersection above it")
        lam: FloatProperty(
            name="lambda", default=1.0, min=0.05, max=6.0,
            description="Enneper-cousin parameter (only its modulus "
                        "matters, so it is taken real and positive)")
        degree: IntProperty(
            name="Degree n", default=1, min=1, max=6,
            description="r1 = z^n in the Theorem B data; n = 1 "
                        "reproduces Enneper's cousin")
        poly_c: FloatProperty(
            name="r2 = c", default=1.0, min=0.05, max=4.0,
            description="the constant polynomial r2 in the Theorem B "
                        "data")
        trinoid_d0: FloatProperty(
            name="d0", default=0.16, min=0.01, max=1.2,
            description="Modulus of the end at z = 0 (the conical order "
                        "of its metric singularity); with d1 and d_inf "
                        "it must satisfy the unitarizable-monodromy "
                        "condition of Bobenko-Pavlyukevich-Springborn "
                        "Prop. 2, or there is no trinoid to draw")
        trinoid_d1: FloatProperty(
            name="d1", default=0.18, min=0.01, max=1.2,
            description="Modulus of the end at z = 1; set all three "
                        "equal for a symmetric trinoid (embedded below "
                        "0.2332, self-intersecting above)")
        trinoid_dinf: FloatProperty(
            name="d_inf", default=0.20, min=0.01, max=1.2,
            description="Modulus of the end at z = infinity")
        extent: FloatProperty(
            name="Domain Extent", default=1.0, min=0.1, max=3.0,
            description="Scales the parameter domain: larger reaches "
                        "further out towards the sphere at infinity")
        ures: IntProperty(name="U Resolution", default=96, min=8,
                          max=400,
                          description="Mesh divisions along the u "
                                      "parameter")
        vres: IntProperty(name="V Resolution", default=96, min=8,
                          max=400,
                          description="Mesh divisions along the v "
                                      "parameter")
        ode_steps: IntProperty(
            name="ODE Steps", default=64, min=8, max=512,
            description="RK4 steps per ray for the Theorem B "
                        "integration")
        shade_smooth: BoolProperty(
            name="Smooth Shading", default=True,
            description="Shade the surface smooth rather than faceted")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the result")

        def execute(self, context):
            mu = self.mu
            if self.mode == 'CATENOID_COUSIN' and abs(mu) < 1e-3:
                # mu = 0 kills the off-diagonal entry entirely and the
                # surface collapses to a line; it is excluded in Bryant.
                self.report({'WARNING'},
                            "mu = 0 is degenerate; nudged to -0.05")
                mu = -0.05
            if self.mode == 'TRINOID':
                # refuse inadmissible moduli rather than draw a
                # plausible wrong surface (BPS Theorem 6 / Prop. 2)
                ok, why = _trinoid.admissible(
                    self.trinoid_d0, self.trinoid_d1, self.trinoid_dinf)
                if not ok:
                    self.report({'ERROR'}, f"Not a trinoid: {why}")
                    return {'CANCELLED'}
            verts, faces, info = build_surface(
                self.mode, self.ures, self.vres, self.model, mu,
                self.lam, self.degree, self.poly_c, self.extent,
                self.ode_steps, self.scale, d0=self.trinoid_d0,
                d1=self.trinoid_d1, dinf=self.trinoid_dinf)
            label = dict(CATENOID_COUSIN="Catenoid Cousin",
                         ENNEPER_COUSIN="Enneper Cousin",
                         POLYNOMIAL="Bryant Surface",
                         TRINOID="CMC-1 Trinoid")[self.mode]
            me = bpy.data.meshes.new(label)
            me.from_pydata([tuple(v) for v in verts], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.shade_smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(label, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            extra = ""
            if self.mode == 'CATENOID_COUSIN':
                extra = (f", total curvature "
                         f"{-4.0 * math.pi * (2.0 * mu + 1.0):.4f}")
            self.report(
                {'INFO'},
                f"{label} ({self.model.title()}): V={len(me.vertices)} "
                f"F={len(me.polygons)}, max |det f - 1| = "
                f"{info['det_dev']:.2e}{extra}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            lay.prop(self, 'model')
            if self.mode == 'CATENOID_COUSIN':
                lay.prop(self, 'mu')
                if abs(self.mu) < 1e-3:
                    lay.label(text="mu = 0 is degenerate", icon='ERROR')
            if self.mode == 'ENNEPER_COUSIN':
                lay.prop(self, 'lam')
            if self.mode == 'POLYNOMIAL':
                lay.prop(self, 'degree')
                lay.prop(self, 'poly_c')
                lay.prop(self, 'ode_steps')
            if self.mode == 'TRINOID':
                lay.prop(self, 'trinoid_d0')
                lay.prop(self, 'trinoid_d1')
                lay.prop(self, 'trinoid_dinf')
                ok, why = _trinoid.admissible(
                    self.trinoid_d0, self.trinoid_d1, self.trinoid_dinf)
                if not ok:
                    lay.label(text=why[:64], icon='ERROR')
            lay.prop(self, 'extent')
            lay.prop(self, 'ures')
            lay.prop(self, 'vres')
            lay.prop(self, 'shade_smooth')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.bryant_surface_add",
                             icon='MESH_UVSPHERE')

    ADD_MENU = True    # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_bryant_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_bryant_surface_add)


def _selftest():
    ok_all = True
    rng = np.random.default_rng(20260819)

    # 1) The immersion lands EXACTLY on the hyperboloid.  For Example 2
    #    this is an algebraic identity --
    #    (mu+1)^4 + mu^4 - 2 mu^2 (mu+1)^2 = (2mu+1)^2 -- so any drift is
    #    a bug, not a tolerance.  Examples 1 and B rely instead on
    #    det F = 1 being preserved, which for B is a genuine test of the
    #    RK4 integration (the Sl(2,C) constraint is not imposed, it is a
    #    conserved quantity of the flow).
    cases = [('CATENOID_COUSIN', dict(mu=-0.3), 1e-13),
             ('CATENOID_COUSIN', dict(mu=1.4), 1e-13),
             ('ENNEPER_COUSIN', dict(lam=1.0), 1e-13),
             ('ENNEPER_COUSIN', dict(lam=2.5), 1e-13),
             ('POLYNOMIAL', dict(degree=1, c=1.0), 1e-9),
             ('POLYNOMIAL', dict(degree=3, c=0.8), 1e-9),
             # trinoid: hypergeometric machinery, det preserved exactly
             ('TRINOID', dict(d0=0.16, d1=0.18, dinf=0.20), 1e-10),
             ('TRINOID', dict(d0=0.2332, d1=0.2332, dinf=0.2332), 1e-10)]
    for mode, kw, tol in cases:
        (u0, u1), (v0, v1), _ = DOMAINS[mode]
        u = rng.uniform(u0, u1, 300)
        v = rng.uniform(v0, v1, 300)
        X = surface_points(mode, u, v, **kw)
        dev = float(hyperboloid_residual(X).max())
        ok = dev < tol and bool((X[:, 0] > 0).all())
        ok_all = ok_all and ok
        print(f"on H^3 {mode:16s} {str(kw):26s}: relative |<x,x>+1| = "
              f"{dev:.2e} (tol {tol:.0e}) {'OK' if ok else 'BAD'}")

    # 2) THE gate: mean curvature ONE, measured in H^3 rather than in the
    #    drawing model.  This is what makes them Bryant surfaces; it is
    #    computed from the Minkowski first and second fundamental forms
    #    and shares no algebra with the construction.
    for mode, kw in (('CATENOID_COUSIN', dict(mu=-0.3)),
                     ('CATENOID_COUSIN', dict(mu=-0.45)),
                     ('CATENOID_COUSIN', dict(mu=0.7)),
                     ('CATENOID_COUSIN', dict(mu=2.0)),
                     ('ENNEPER_COUSIN', dict(lam=1.0)),
                     ('ENNEPER_COUSIN', dict(lam=2.0)),
                     ('POLYNOMIAL', dict(degree=1, c=1.0)),
                     ('POLYNOMIAL', dict(degree=2, c=1.0)),
                     ('POLYNOMIAL', dict(degree=3, c=0.8)),
                     ('TRINOID', dict(d0=0.16, d1=0.18, dinf=0.20)),
                     ('TRINOID', dict(d0=0.16, d1=0.18, dinf=0.20,
                                      domain=0)),
                     ('TRINOID', dict(d0=0.16, d1=0.18, dinf=0.20,
                                      domain=2))):
        (u0, u1), (v0, v1), _ = DOMAINS[mode]
        # stay off the domain edges, where the FD stencil would leave it
        u = rng.uniform(0.75 * u0, 0.75 * u1, 120)
        v = rng.uniform(0.75 * v0, 0.75 * v1, 120)
        Hm = np.abs(mean_curvature_h3(mode, u, v, **kw))
        Hm = Hm[np.isfinite(Hm)]
        med = float(np.median(Hm))
        q1, q3 = np.percentile(Hm, [25.0, 75.0])
        ok = abs(med - 1.0) < 2e-3 and (q3 - q1) < 1e-2
        ok_all = ok_all and ok
        print(f"|H| in H^3 {mode:16s} {str(kw):22s}: median {med:.6f} "
              f"(want 1) IQR {q3 - q1:.1e} {'OK' if ok else 'BAD'}")

    # 3) Example 2 is a surface of REVOLUTION about the x3 axis, and is
    #    symmetric under r -> 1/r (which swaps f00 and f11, i.e. flips
    #    x3).  Both are exact, not approximate.
    s = rng.uniform(-2.0, 2.0, 200)
    th = rng.uniform(0.0, 2 * math.pi, 200)
    A = catenoid_cousin(s, th, -0.3)
    B = catenoid_cousin(s, th + 0.7, -0.3)
    rot_ok = float(np.abs(np.hypot(A[:, 1], A[:, 2])
                          - np.hypot(B[:, 1], B[:, 2])).max())
    C = catenoid_cousin(-s, th, -0.3)
    refl = float(max(np.abs(A[:, 0] - C[:, 0]).max(),
                     np.abs(A[:, 3] + C[:, 3]).max()))
    ok = rot_ok < 1e-12 and refl < 1e-12
    ok_all = ok_all and ok
    print(f"symmetry Example 2: revolution {rot_ok:.1e}, r->1/r "
          f"reflection {refl:.1e} {'OK' if ok else 'BAD'}")

    # 4) The catenoid cousin's profile is EMBEDDED for -1/2 < mu < 0 and
    #    has exactly one self-intersection for mu > 0 (Bryant, p. 342).
    #    In the half-plane the profile is (rho, x3) with
    #    rho = sqrt(x1^2 + x2^2); an embedded profile is monotone in x3,
    #    a once-crossing one is not.
    for mu, want_embedded in ((-0.45, True), (-0.3, True), (-0.1, True),
                              (0.3, False), (1.0, False), (2.5, False)):
        s = np.linspace(-3.0, 3.0, 4000)
        Xp = catenoid_cousin(s, np.zeros_like(s), mu)
        x3 = Xp[:, 3]
        d = np.diff(x3)
        monotone = bool((d > 0).all() or (d < 0).all())
        ok = monotone == want_embedded
        ok_all = ok_all and ok
        print(f"profile mu={mu:+.2f}: x3 monotone {str(monotone):5s} "
              f"-> {'embedded' if monotone else 'self-intersecting'} "
              f"(want {'embedded' if want_embedded else 'crossing'}) "
              f"{'OK' if ok else 'BAD'}")

    # 5) Theorem B at degree 1 IS Enneper's cousin: the potential of
    #    Example 1 is A with (r1, r2) = (i lam z, i lam), so
    #    r1 = z, r2 = 1 must reproduce lam = 1 up to an isometry of H^3.
    #    Isometries preserve hyperbolic distance, so compare the induced
    #    metric rather than the coordinates.
    xs = rng.uniform(-1.0, 1.0, 200)
    ys = rng.uniform(-1.0, 1.0, 200)
    hh = 1e-4
    E1 = enneper_cousin(xs, ys, 1.0)
    E2 = polynomial_bryant(xs, ys, 1, 1.0, steps=128)
    dE1 = (enneper_cousin(xs + hh, ys, 1.0) - E1) / hh
    dE2 = (polynomial_bryant(xs + hh, ys, 1, 1.0, steps=128) - E2) / hh
    g1, g2 = minkowski(dE1, dE1), minkowski(dE2, dE2)
    rel = float(np.abs(g1 - g2).max() / np.abs(g1).max())
    ok = rel < 5e-3
    ok_all = ok_all and ok
    print(f"Theorem B(n=1) == Enneper cousin: induced metric agrees to "
          f"{rel:.2e} {'OK' if ok else 'BAD'}")

    # 6) meshes build in every model, are finite, not collapsed, and the
    #    ball models really are inside the unit ball
    for mode in MODES:
        for model in MODELS:
            P, F, info = build_surface(mode, 40, 40, model, steps=32)
            P = np.asarray(P)
            ok = len(F) > 0 and np.isfinite(P).all()
            ok_all = ok_all and ok
            print(f"build {mode:16s} {model:11s}: V={len(P)} F={len(F)} "
                  f"det dev {info['det_dev']:.1e} "
                  f"{'OK' if ok else 'BAD'}")
    # unit-ball containment, checked before the mesh normalisation
    for mode in MODES:
        (u0, u1), (v0, v1), _ = DOMAINS[mode]
        u = rng.uniform(u0, u1, 400)
        v = rng.uniform(v0, v1, 400)
        X = surface_points(mode, u, v)
        rad = np.linalg.norm(to_model(X, 'POINCARE'), axis=-1)
        ok = float(rad.max()) < 1.0
        ok_all = ok_all and ok
        print(f"Poincare ball {mode:16s}: max radius {rad.max():.6f} "
              f"(must be < 1) {'OK' if ok else 'BAD'}")

    assert ok_all
    print("bryant surface standalone tests passed")
