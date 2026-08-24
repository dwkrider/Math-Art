
# Swept Surfaces generator for Blender: surfaces traced by a curve
# moving through space.
#
# The organising idea is DARBOUX's: a surface swept by a rigid curve --
# kinematically, the trace of an unbending wire carried through space --
# is a union of congruent copies of that curve, and the classical
# families are all special cases of the one construction.  Translate the
# wire and you get a translation surface; spin it about a fixed axis and
# you get a surface of revolution; screw it and you get a helicoid.  A
# general rigid motion gives none of the three, and that is what the
# DARBOUX mode builds: the wire spins about the axis while tumbling, and
# its centre rides a circle.  The mathematics and the congruence gate
# live in `math_art/surfaces/encyclopedia.py`.
#
# The other three modes are fixed formulas rather than motions.  They
# are what this module shipped before Darboux gave it a general case,
# and they keep their own parameters:
#
#   HYPERBOLIC_HELICOID -- the helicoid's hyperbolic cousin,
#     a twisted band whose points all share the denominator
#     1 + cosh(u) cosh(v); the torsion tau sets how fast the
#     band winds around its axis.
#   SEASHELL -- a conical spiral shell: a circular tube whose
#     radius shrinks linearly to an apex while it winds n
#     whorls around the axis and rises by a fixed height.
#   CORKSCREW -- the "twisted sphere": a sphere stretched
#     along a diameter and sheared upward while it turns, so
#     each meridian circle climbs by b per radian of turn.
#   HELICONE -- the helico-conical surface, where the curve is carried
#     by SIMILARITIES rather than rigid motions: turned and scaled at
#     once, so its copies are similar but not congruent and each
#     u-curve is a conical spiral.  It is the one sweep here that is
#     deliberately not a Darboux surface.
#   EGG_BOX -- z = a(sin(x/b) + sin(y/b)), one sinusoid slid along a
#     perpendicular one: the simplest translation surface, and a
#     standard picture of extrema and saddle points.
#   SINE_TORUS -- a torus whose tube breathes, z = b sin(v) cos(k u).
#     One parameter gives three surfaces: integer k a torus,
#     half-odd-integer k a KLEIN BOTTLE (the tube returns inside out
#     after one turn), and k = 1/2 with equal radii a CROSS-CAP, the
#     closing of the hole dropping the genus by one.  El-Milick built
#     that model in 1947 and called it a one-sided cyclide.
#
# The ROTOID motion added to the Darboux family carries the curve along
# a helical SPINE instead of about a fixed axis, turning it as it goes:
# no turn gives a tube (a coil, or a torus if the spine is flat) and a
# turn gives the generalized helicoid proper.
#
# Each surface is built by a pure-python function that works
# without bpy, so this file can be run standalone for its
# self-tests.  Seams where a parameter wraps (the seashell's
# tube direction, the corkscrew's meridian circles) are welded
# by index so the meshes are seamless; the seashell's apex,
# where the whole tube collapses to one point, is emitted as a
# single vertex with a triangle fan.
#
# References:
# - G. Darboux, "Lecons sur la theorie generale des surfaces", 1887-96
#   -- the surfaces swept by a rigid curve.  The classification of the
#   three special motions followed here is from R. Ferreol,
#   "Encyclopedie des formes mathematiques remarquables", mathcurve.com,
#   chapter "surface de Darboux"; a converted copy is in
#   research/books/mathcurve_encyclopedie_formes_mathematiques/.
# - Rotoid, helico-conical surface, egg box and sine torus: R. Ferreol,
#   "Encyclopedie des formes mathematiques remarquables"
#   (mathcurve.com), chapters "rotoide", "surface helicoconique",
#   "boite a oeufs" and "tore sinusoidal".
# - Sine torus at k = 1/2: Maurice El-Milick (1947), who called it a
#   one-sided cyclide; his model is in the Institut Henri Poincare
#   collection.
# - The helicoid is a classical minimal and ruled surface
#   (J. B. C. Meusnier, 1776); the hyperbolic helicoid, conical
#   seashell and twisted-sphere forms here are standard parametric
#   surfaces. See A. Gray, E. Abbena, S. Salamon, "Modern
#   Differential Geometry of Curves and Surfaces with Mathematica"
#   (3rd ed., 2006), and J. Meier's gallery (3d-meier.de).

bl_info = {
    "name": "Swept Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Darboux surfaces swept by a rigid curve, plus "
                   "the hyperbolic helicoid, seashell and corkscrew",
    "category": "Add Mesh",
}

import math
from fractions import Fraction

import numpy as np

_TWO_PI = 2.0 * math.pi

try:
    from .surfaces.encyclopedia import build_darboux
except ImportError:                       # flat import outside the package
    from surfaces.encyclopedia import build_darboux

try:
    from . import rim_curve as _rim
except ImportError:  # flat import outside the package
    import rim_curve as _rim

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty,
                           EnumProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def build_hyperbolic_helicoid(tau=2.5, extent=4.0, res=96):
    """(verts, faces): the hyperbolic helicoid

        x = sinh(v) cos(tau u) / (1 + cosh(u) cosh(v))
        y = sinh(v) sin(tau u) / (1 + cosh(u) cosh(v))
        z = cosh(v) sinh(u) / (1 + cosh(u) cosh(v))

    over u, v in [-extent, extent].  An open twisted band: no
    parameter wraps, so there is no seam to weld.  The shared
    denominator is >= 2 everywhere, so every point is finite."""
    nu = nv = max(2, int(res))
    u = np.linspace(-extent, extent, nu + 1)
    v = np.linspace(-extent, extent, nv + 1)
    U, V = np.meshgrid(u, v, indexing='ij')
    den = 1.0 + np.cosh(U) * np.cosh(V)
    X = np.sinh(V) * np.cos(tau * U) / den
    Y = np.sinh(V) * np.sin(tau * U) / den
    Z = np.cosh(V) * np.sinh(U) / den
    verts = [(X[i, j], Y[i, j], Z[i, j])
             for i in range(nu + 1) for j in range(nv + 1)]
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            b = (i + 1) * (nv + 1) + j
            faces.append([a, b, b + 1, a + 1])
    return verts, faces


def build_seashell(whorls=2, aspect=0.2, height=1.0,
                   opening=0.1, res=96):
    """(verts, faces): a conical seashell surface

        r(u, v) = (1 - v/2pi) (1 + cos u) + c
        x = r cos(n v)
        y = r sin(n v)
        z = b v / 2pi + a sin(u) (1 - v/2pi)

    with n = whorls, a = aspect (vertical stretch of the tube
    section), b = height (total rise), c = opening (offset
    radius that keeps the mouth open), u, v in [0, 2pi].  The
    tube direction u wraps and is welded by index; at v = 2pi
    the whole tube collapses to the apex (c cos 2pi n,
    c sin 2pi n, b), emitted once and fanned with triangles."""
    nu = nv = max(3, int(res))
    two_pi = 2.0 * math.pi
    n, a, b, c = float(whorls), aspect, height, opening
    verts = []
    for j in range(nv):            # spiral rows, apex excluded
        v = two_pi * j / nv
        taper = 1.0 - v / two_pi
        for i in range(nu):        # tube ring, u wraps
            u = two_pi * i / nu
            r = taper * (1.0 + math.cos(u)) + c
            verts.append((r * math.cos(n * v),
                          r * math.sin(n * v),
                          b * v / two_pi
                          + a * math.sin(u) * taper))
    apex = len(verts)
    verts.append((c * math.cos(two_pi * n),
                  c * math.sin(two_pi * n), b))
    faces = []
    for j in range(nv - 1):
        for i in range(nu):
            i2 = (i + 1) % nu
            faces.append([j * nu + i, j * nu + i2,
                          (j + 1) * nu + i2, (j + 1) * nu + i])
    last = (nv - 1) * nu
    for i in range(nu):
        faces.append([last + i, last + (i + 1) % nu, apex])
    return verts, faces


def build_corkscrew(a=1.0, b=0.3, res=96):
    """(verts, faces): the corkscrew surface (twisted sphere)

        x = a cos(u) cos(v)
        y = a sin(u) cos(v)
        z = a sin(v) + b u

    with a the sphere radius and b the axial rise per radian
    of twist, u, v in [0, 2pi].  Each meridian circle (fixed
    u, v wrapping) is welded by index; u is left open because
    the twist offsets its two ends by 2 pi b in z."""
    nu = nv = max(3, int(res))
    two_pi = 2.0 * math.pi
    verts = []
    for i in range(nu + 1):        # twist direction, open
        u = two_pi * i / nu
        for j in range(nv):        # meridian circle, v wraps
            v = two_pi * j / nv
            verts.append((a * math.cos(u) * math.cos(v),
                          a * math.sin(u) * math.cos(v),
                          a * math.sin(v) + b * u))
    faces = []
    for i in range(nu):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append([i * nv + j, i * nv + j2,
                          (i + 1) * nv + j2, (i + 1) * nv + j])
    return verts, faces


def build_helicone(profile=2, turns=2.0, height=1.0, res_u=64, res_v=160):
    """(verts, faces): the helico-conical surface, or "helicone".

        x = v (f(u) cos v - g(u) sin v)
        y = v (f(u) sin v + g(u) cos v)
        z = v h(u)

    The generatrix is carried by SIMILARITIES -- rotate by v and scale
    by v at the same time -- so the copies are all similar to one
    another but, unlike a Darboux sweep, not congruent: they grow as
    they turn.  Each u = constant curve is therefore a conical spiral,
    and the surface fills a cone.

    `profile` picks the generatrix: 2 is the parabola of the source's
    figure, higher values are the flatter even powers.
    """
    nu, nv = max(3, int(res_u)), max(3, int(res_v))
    p = max(1, int(profile))
    u = np.linspace(-1.0, 1.0, nu)
    f, g, h = u, np.zeros_like(u), height * (1.0 - np.abs(u) ** p)
    v = np.linspace(0.05, _TWO_PI * turns, nv)
    cv, sv = np.cos(v), np.sin(v)
    P = np.stack([
        v[:, None] * (f[None, :] * cv[:, None] - g[None, :] * sv[:, None]),
        v[:, None] * (f[None, :] * sv[:, None] + g[None, :] * cv[:, None]),
        v[:, None] * h[None, :] * np.ones_like(cv)[:, None]], -1)
    verts = [tuple(P[i, j]) for i in range(nv) for j in range(nu)]
    faces = [[i * nu + j, i * nu + j + 1,
              (i + 1) * nu + j + 1, (i + 1) * nu + j]
             for i in range(nv - 1) for j in range(nu - 1)]
    return verts, faces


def build_translation_surface(amp=0.35, wavelength=1.0, extent=3.0,
                              res=72):
    """(verts, faces): the EGG BOX, z = a (sin(x/b) + sin(y/b)).

    A translation surface is the sum of two curves, P(u,v) = c1(u) +
    c2(v) -- one curve slid along the other without turning.  Take both
    to be the same sinusoid, laid perpendicular, and the sum is the egg
    box: a field of alternating bumps and hollows whose saddle points
    sit where one curve is rising as fast as the other falls.

    Ferreol notes the curiosity that replacing the sum by a PRODUCT
    gives an egg box too, because
    a sin(x/b) sin(y/b) is itself a sum of two sinusoids in the rotated
    coordinates (y-x)/b and (x+y)/b -- the same surface turned 45
    degrees and rescaled.  `_selftest` checks that identity.
    """
    n = max(3, int(res))
    b = max(1e-6, wavelength)
    t = np.linspace(-extent, extent, n)
    c1 = np.stack([t, np.zeros_like(t), amp * np.sin(t / b)], -1)
    c2 = np.stack([np.zeros_like(t), t, amp * np.sin(t / b)], -1)
    P = c1[:, None, :] + c2[None, :, :]
    verts = [tuple(P[i, j]) for i in range(n) for j in range(n)]
    faces = [[i * n + j, i * n + j + 1, (i + 1) * n + j + 1,
              (i + 1) * n + j]
             for i in range(n - 1) for j in range(n - 1)]
    return verts, faces


def build_sine_torus(major=1.0, minor=0.4, k=0.5, res_u=192, res_v=64):
    """(verts, faces): the SINE TORUS,

        x = (a + b cos v) cos u,  y = (a + b cos v) sin u,
        z = b sin v cos(k u)

    A torus whose tube, instead of being circular, is an ellipse one of
    whose axes breathes sinusoidally as it goes round -- so the source
    describes it as a variable ellipse revolved about an axis.  The
    single parameter k decides what it is:

        k = 0                  z no longer depends on u: an exact TORUS
        k = 1/2 and a > b      a KLEIN BOTTLE, immersed in 3-space
        k = 1/2 and a = b      the hole closes and it becomes a
                               CROSS-CAP -- losing the hole drops the
                               genus by one, carrying the Klein bottle
                               to the projective plane

    El-Milick built the k = 1/2 model in 1947 and called it a one-sided
    cyclide; it is in the Institut Henri Poincare collection.

    What closes the surface is the seam, and getting it wrong silently
    turns every case into a torus.  After ONE turn in u,

        z(u + 2pi, v) = b sin v cos(ku + 2 pi k)

    so for integer k the surface simply meets itself -- a torus seam --
    while for half-odd-integer k the sign flips and it meets itself
    REVERSED: F(u + 2pi, v) = F(u, -v).  The tube comes back inside out,
    which is exactly the Klein bottle's identification.  So the grid runs
    one turn and its seam is joined by INDEX with v negated in the
    flipped case.  Welding by coordinate instead would fuse the
    self-intersections an immersion necessarily has and destroy the
    Euler characteristic; wrapping both directions plainly (the obvious
    thing) forces chi = 0 and orientable no matter what the geometry
    does.  A ring that genuinely degenerates to a point -- which is how
    a = b closes the hole -- is collapsed to a single vertex.
    """
    frac = Fraction(float(k)).limit_denominator(12)
    flip = (2 * frac).denominator == 1 and int(2 * frac) % 2 == 1
    turns = 1 if flip else frac.denominator
    nu = max(8, int(res_u))
    nv = max(6, int(res_v))
    nv += nv % 2                      # v -> -v must pair up columns
    u = _TWO_PI * turns * np.arange(nu) / nu
    v = _TWO_PI * np.arange(nv) / nv
    R = major + minor * np.cos(v)
    P = np.stack([
        R[None, :] * np.cos(u)[:, None],
        R[None, :] * np.sin(u)[:, None],
        (minor * np.sin(v))[None, :] * np.cos(k * u)[:, None]], -1)

    index = np.arange(nu * nv).reshape(nu, nv)
    keep = np.ones(nu * nv, dtype=bool)
    for j in range(nv):                       # collapse degenerate rings
        ring = P[:, j, :]
        if float(np.max(np.abs(ring - ring[0]))) < 1e-12:
            keep[index[1:, j]] = False
            index[1:, j] = index[0, j]
    order = np.cumsum(keep) - 1
    verts = [tuple(P.reshape(-1, 3)[i]) for i in range(nu * nv) if keep[i]]

    def vid(i, j):
        if i >= nu:                   # across the seam
            i -= nu
            if flip:
                j = -j
        return int(order[index[i, j % nv]])
    faces = []
    for i in range(nu):
        for j in range(nv):
            ring = [vid(i, j), vid(i + 1, j),
                    vid(i + 1, j + 1), vid(i, j + 1)]
            clean = [ring[0]] + [b for a, b in zip(ring, ring[1:])
                                 if a != b]
            if len(clean) > 2 and clean[0] == clean[-1]:
                clean = clean[:-1]
            if len(clean) >= 3:
                faces.append(clean)
    return verts, faces


_SURFACES = [
    ('DARBOUX', "Darboux Surface",
     "A rigid curve swept by a motion. Translation, revolution and "
     "screw motions give translation surfaces, surfaces of revolution "
     "and helicoids; a general motion gives a Darboux surface that is "
     "none of the three"),
    ('HYPERBOLIC_HELICOID', "Hyperbolic Helicoid",
     "Twisted band with torsion tau over the shared "
     "denominator 1 + cosh(u) cosh(v)"),
    ('SEASHELL', "Seashell",
     "Conical spiral shell winding n whorls to an apex"),
    ('CORKSCREW', "Corkscrew",
     "Twisted sphere: a sphere sheared upward as it turns"),
    ('HELICONE', "Helico-Conical Surface",
     "A curve carried by SIMILARITIES -- turned and scaled together -- "
     "so its copies are similar rather than congruent and each traces "
     "a conical spiral"),
    ('EGG_BOX', "Egg Box",
     "z = a(sin(x/b) + sin(y/b)): one sinusoid slid along a "
     "perpendicular one, the simplest translation surface"),
    ('SINE_TORUS', "Sine Torus",
     "A torus whose tube breathes: z = b sin(v) cos(k u). Integer k "
     "gives a torus, half-odd-integer k a Klein bottle, and k = 1/2 "
     "with equal radii a cross-cap"),
]


if _IN_BLENDER:

    class MESH_OT_helical_surface_add(bpy.types.Operator):
        """Add a surface swept by a curve moving through space: a
        Darboux surface (a rigid curve carried by a motion, of which
        translation surfaces, surfaces of revolution and helicoids are
        the special cases), or one of three classic helical formulas --
        the hyperbolic helicoid, a conical seashell, or the corkscrew
        surface (twisted sphere)"""
        # The id stays `helical_surface_add`: renaming it would break
        # every script, menu entry and subject table that names it, and
        # the label is what the user actually reads.
        bl_idname = "mesh.helical_surface_add"
        bl_label = "Swept Surface"
        bl_options = {'REGISTER', 'UNDO'}
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()
        rim_profile: _rim.rim_profile_prop()
        rim_twist: _rim.rim_twist_prop()
        rim_reeds: _rim.rim_reeds_prop()

        surface: EnumProperty(
            name="Surface", items=_SURFACES,
            default='HYPERBOLIC_HELICOID',
            description="Which swept surface to build")
        motion: EnumProperty(
            name="Motion",
            items=[('GENERAL', "General Darboux Motion",
                    "Spin about the axis while tumbling, with the "
                    "curve's centre riding a circle: neither a "
                    "translation surface, nor a surface of revolution, "
                    "nor a helicoid"),
                   ('TRANSLATION', "Translation",
                    "Slide the curve along a straight line"),
                   ('REVOLUTION', "Revolution",
                    "Spin the curve about a fixed axis"),
                   ('HELICOID', "Helicoidal",
                    "Screw the curve about a fixed axis")],
            default='GENERAL',
            description="How the rigid curve is carried through space "
                        "(Darboux surface only)")
        generatrix: EnumProperty(
            name="Generatrix",
            items=[('CIRCLE', "Circle",
                    "A circular generatrix; under a rotation this "
                    "gives a cyclic surface"),
                   ('ELLIPSE', "Ellipse", "An elliptical generatrix"),
                   ('LEMNISCATE', "Gerono Lemniscate",
                    "A figure-eight generatrix"),
                   ('ASTROID', "Astroid",
                    "A four-cusped generatrix"),
                   ('SEGMENT', "Segment",
                    "A straight generatrix, which makes the result a "
                    "ruled surface")],
            default='CIRCLE',
            description="The rigid curve that is swept (Darboux only)")
        curve_size: FloatProperty(
            name="Curve Size", default=0.45, min=0.02, max=5.0,
            description="Size of the swept curve (Darboux only)")
        curve_ratio: FloatProperty(
            name="Curve Ratio", default=0.5, min=0.02, max=5.0,
            description="Minor-to-major ratio of an elliptical "
                        "generatrix (Darboux only)")
        path_radius: FloatProperty(
            name="Path Radius", default=1.0, min=0.0, max=10.0,
            description="Radius of the circle the curve's centre rides "
                        "(Darboux only)")
        pitch: FloatProperty(
            name="Pitch", default=0.4, min=-5.0, max=5.0,
            description="Rise per radian for the translation and screw "
                        "motions (Darboux only)")
        tilt: FloatProperty(
            name="Tumble", default=0.6, min=0.0, max=3.14,
            description="How far the curve tips out of its plane as it "
                        "goes round; 0 collapses the general motion "
                        "back to a plain revolution (Darboux only)")
        wobbles: IntProperty(
            name="Tumbles", default=3, min=1, max=24,
            description="How many times the curve tips back and forth "
                        "per revolution (Darboux only)")
        turns: FloatProperty(
            name="Turns", default=1.0, min=0.05, max=12.0,
            description="How far the motion runs, in revolutions. A "
                        "whole number closes the sweep and welds the "
                        "seam; anything else leaves it open (Darboux "
                        "only)")
        tau: FloatProperty(
            name="Torsion", default=2.5, min=0.0, max=20.0,
            description="Torsion tau of the hyperbolic "
                        "helicoid (turns per unit of the "
                        "spine parameter)")
        extent: FloatProperty(
            name="Extent", default=4.0, min=0.5, max=10.0,
            description="Half-range of both parameters of "
                        "the hyperbolic helicoid")
        whorls: IntProperty(
            name="Whorls", default=2, min=1, max=8,
            description="Number of turns of the shell spiral")
        aspect: FloatProperty(
            name="Tube Aspect", default=1.0, min=0.0, max=2.0,
            description="Vertical stretch of the tube "
                        "cross-section (a)")
        height: FloatProperty(
            name="Height", default=4.0, min=0.0, max=10.0,
            description="Total rise of the shell from mouth "
                        "to apex (b)")
        opening: FloatProperty(
            name="Opening", default=0.37, min=0.0, max=2.0,
            description="Offset radius that keeps the shell "
                        "mouth open (c)")
        profile: IntProperty(
            name="Profile Power", default=2, min=1, max=8,
            description="Generatrix z = 1 - |u|^p; 2 is the parabola of "
                        "the source's figure and higher powers flatten "
                        "it (helico-conical surface only)")
        wavelength: FloatProperty(
            name="Wavelength", default=1.0, min=0.05, max=10.0,
            description="b in sin(x/b): the spacing of the bumps "
                        "(egg box only)")
        sine_k: FloatProperty(
            name="Modulation k", default=0.5, min=0.0, max=4.0,
            description="k in z = b sin(v) cos(k u).  Whole numbers "
                        "give a torus; halves of odd numbers bring the "
                        "tube back inside out and give a KLEIN BOTTLE, "
                        "and k = 1/2 with the two radii equal closes "
                        "the hole into a CROSS-CAP (sine torus only)")
        cork_a: FloatProperty(
            name="Sphere Radius", default=0.5, min=0.01,
            max=10.0,
            description="Radius a of the twisted sphere")
        cork_b: FloatProperty(
            name="Twist Rise", default=0.3, min=0.0, max=5.0,
            description="Axial rise b per radian of twist")
        resolution: IntProperty(
            name="Resolution", default=96, min=8, max=512,
            description="Segments along each parametric "
                        "direction")
        smooth: BoolProperty(name="Smooth Shading",
                             default=True,
                             description="Shade the surface smooth")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="Solidify modifier thickness (0 = raw "
                        "surface)")
        scale: FloatProperty(name="Scale", default=1.0,
                             min=0.01, max=100.0,
                             description="Overall size of the result")

        def execute(self, context):
            if self.surface == 'DARBOUX':
                V, faces = build_darboux(
                    motion=self.motion, generatrix=self.generatrix,
                    size=self.curve_size, ratio=self.curve_ratio,
                    radius=self.path_radius, pitch=self.pitch,
                    tilt=self.tilt, wobbles=self.wobbles,
                    turns=self.turns, res_u=self.resolution,
                    res_v=2 * self.resolution)
                verts = [tuple(map(float, v)) for v in V]
                name = "Darboux Surface"
            elif self.surface == 'HYPERBOLIC_HELICOID':
                verts, faces = build_hyperbolic_helicoid(
                    self.tau, self.extent, self.resolution)
                name = "Hyperbolic Helicoid"
            elif self.surface == 'SEASHELL':
                verts, faces = build_seashell(
                    self.whorls, self.aspect, self.height,
                    self.opening, self.resolution)
                name = "Seashell"
            elif self.surface == 'HELICONE':
                verts, faces = build_helicone(
                    self.profile, self.turns, self.height,
                    self.resolution, 2 * self.resolution)
                name = "Helico-Conical Surface"
            elif self.surface == 'EGG_BOX':
                verts, faces = build_translation_surface(
                    self.aspect, self.wavelength, self.extent,
                    self.resolution)
                name = "Egg Box"
            elif self.surface == 'SINE_TORUS':
                verts, faces = build_sine_torus(
                    self.path_radius, self.curve_size, self.sine_k,
                    3 * self.resolution, self.resolution)
                name = "Sine Torus"
            else:
                verts, faces = build_corkscrew(
                    self.cork_a, self.cork_b, self.resolution)
                name = "Corkscrew"
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) \
                or 1.0
            s = self.scale / half
            verts = [tuple((v[k] - (lo[k] + hi[k]) / 2.0) * s
                           for k in range(3)) for v in verts]
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set(
                'use_smooth', [self.smooth] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            if self.rim:
                _ob = context.active_object
                if _ob is not None:
                    _rim.add_rim_from_object(
                        context, _ob, _ob.name,
                        self.rim_thickness, self.rim_smooth,
                        self.rim_profile, twist=self.rim_twist,
                        reeds=self.rim_reeds)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'surface')
            if self.surface == 'DARBOUX':
                keys = ('motion', 'generatrix', 'curve_size')
                if self.generatrix == 'ELLIPSE':
                    keys += ('curve_ratio',)
                if self.motion in ('GENERAL', 'TRANSLATION'):
                    keys += ('path_radius',)
                if self.motion in ('TRANSLATION', 'HELICOID'):
                    keys += ('pitch',)
                if self.motion == 'GENERAL':
                    keys += ('tilt', 'wobbles')
                keys += ('turns',)
            elif self.surface == 'HYPERBOLIC_HELICOID':
                keys = ('tau', 'extent')
            elif self.surface == 'SEASHELL':
                keys = ('whorls', 'aspect', 'height',
                        'opening')
            elif self.surface == 'HELICONE':
                keys = ('profile', 'turns', 'height')
            elif self.surface == 'EGG_BOX':
                keys = ('aspect', 'wavelength', 'extent')
            elif self.surface == 'SINE_TORUS':
                keys = ('path_radius', 'curve_size', 'sine_k')
            else:
                keys = ('cork_a', 'cork_b')
            for k in keys + ('resolution', 'smooth',
                             'thickness', 'scale'):
                lay.prop(self, k)

            _rim.draw_rim(lay, self)
    def _menu_func(self, context):
        self.layout.operator("mesh.helical_surface_add",
                             icon='MOD_SCREW')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_helical_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_helical_surface_add)


def _selftest():
    builds = [
        ("hyperbolic helicoid",
         lambda: build_hyperbolic_helicoid(res=48)),
        ("seashell", lambda: build_seashell(res=48)),
        ("corkscrew", lambda: build_corkscrew(res=48)),
    ]
    for label, fn in builds:
        verts, faces = fn()
        assert len(verts) > 0 and len(faces) > 0
        finite = all(all(math.isfinite(c) for c in v)
                     for v in verts)
        assert finite, f"{label}: non-finite coordinate"
        valid = all(0 <= i < len(verts)
                    for f in faces for i in f)
        assert valid, f"{label}: face index out of range"
        print(f"{label}: V={len(verts)} F={len(faces)} "
              f"finite={finite} indices ok={valid}")
    # the seashell apex (last vertex, where the tube
    # collapses) must be a clean finite point
    verts, faces = build_seashell(res=48)
    apex = verts[-1]
    assert all(math.isfinite(c) for c in apex)
    assert not any(math.isnan(c) for c in apex)
    # every apex triangle references the single apex vertex
    tris = [f for f in faces if len(f) == 3]
    assert len(tris) == 48
    assert all(f[2] == len(verts) - 1 for f in tris)
    print(f"seashell apex {tuple(round(c, 6) for c in apex)}"
          f" welded into {len(tris)} triangles")
    # ---- helicone: similar copies, not congruent ones -----------------
    # A Darboux sweep carries a RIGID curve, so its copies are congruent.
    # The helicone carries it by similarities instead, so distances
    # inside a copy all scale by the same factor v -- checking the whole
    # distance matrix says "similar" and would catch a shear that a
    # single distance would not.
    nu, nv = 24, 40
    V, F = build_helicone(res_u=nu, res_v=nv)
    P = np.asarray(V).reshape(nv, nu, 3)
    ratios = []
    for row in (5, 17, 33):
        D0 = np.linalg.norm(P[5][:, None] - P[5][None, :], axis=-1)
        Dk = np.linalg.norm(P[row][:, None] - P[row][None, :], axis=-1)
        m = D0 > 1e-9
        ratios.append(float((Dk[m] / D0[m]).std()))
    assert max(ratios) < 1e-9, ratios
    print("helicone: each swept copy is SIMILAR to the first (distance "
          "ratios constant to %.0e), not congruent -- a similarity "
          "sweep, unlike Darboux's" % max(ratios))

    # ---- egg box: the sum/product identity ---------------------------
    # a sin(x/b) sin(y/b) = (a/2)(sin((y-x+pi/2)/b) + sin((x+y-pi/2)/b)),
    # so the product form really is the same surface turned and rescaled.
    xs = np.linspace(-3.0, 3.0, 41)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    b = 1.0
    lhs = np.sin(X / b) * np.sin(Y / b)
    rhs = 0.5 * (np.sin((Y - X + math.pi / 2) / b)
                 + np.sin((X + Y - math.pi / 2) / b))
    assert float(np.max(np.abs(lhs - rhs))) < 1e-12
    V, F = build_translation_surface(res=24)
    P = np.asarray(V).reshape(24, 24, 3)
    # a translation surface: sliding along v is the same shift at every u
    shift = P[:, 7, :] - P[:, 3, :]
    assert float(np.max(np.abs(shift - shift[0]))) < 1e-12
    print("egg box: sum and product forms agree to 1e-12, and the sweep "
          "is a pure translation (same shift at every station)")

    # ---- sine torus: one parameter, three surfaces --------------------
    # The seam is what decides the topology, so each case is checked on
    # chi AND sidedness -- chi = 0 alone cannot tell a torus from a
    # Klein bottle.
    try:
        from .minsurf.topology import euler_characteristic as _euler
        from .other_polyhedra_generator import is_orientable as _ori
    except ImportError:                       # flat import
        from minsurf.topology import euler_characteristic as _euler
        from other_polyhedra_generator import is_orientable as _ori
    cases = ((1.0, 0.4, 0.0, 0, True, "torus"),
             (1.0, 0.4, 1.0, 0, True, "torus"),
             (1.0, 0.4, 0.5, 0, False, "Klein bottle"),
             (1.0, 0.4, 1.5, 0, False, "Klein bottle"),
             (1.0, 1.0, 0.5, 1, False, "cross-cap"))
    for a, bb, kk, chi_w, ori_w, label in cases:
        V, F = build_sine_torus(a, bb, kk, res_u=180, res_v=42)
        F = [list(f) for f in F]
        cnt = {}
        for f in F:
            for i in range(len(f)):
                p, q = f[i], f[(i + 1) % len(f)]
                cnt[(min(p, q), max(p, q))] = \
                    cnt.get((min(p, q), max(p, q)), 0) + 1
        assert all(c == 2 for c in cnt.values()), (label, "not closed")
        chi = _euler(len(V), F)
        ori = _ori(F)
        assert chi == chi_w and ori == ori_w, (label, chi, ori)
    # and k = 0 is an EXACT torus, not merely a torus-shaped thing
    V, F = build_sine_torus(1.0, 0.4, 0.0, res_u=96, res_v=48)
    A = np.asarray(V)
    d = np.hypot(np.hypot(A[:, 0], A[:, 1]) - 1.0, A[:, 2])
    assert float(d.max() - d.min()) < 1e-12, (d.min(), d.max())
    print("sine torus: k = 0 and 1 give a torus (k = 0 exactly, tube "
          "radius constant to 1e-12), k = 1/2 and 3/2 a Klein bottle, "
          "and k = 1/2 with a = b a cross-cap")

    print("helical standalone tests passed")
