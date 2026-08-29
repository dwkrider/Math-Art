"""Parametric charts for generator-defined records -- verified against
the shipped builders themselves.

`charts.py` covers the records whose curvature condition (minimal, flat,
K = +-1) can gate a transcription.  Most parametric records have no such
condition: a mistranscribed seashell is just a different seashell, and
nothing intrinsic would catch it.  What those records DO have is an
authoritative definition already in the repository -- the shipped
builder named in their `construction` block, which is pure numpy and
imports without Blender.  So here the gate is reproduction: every chart
below is sampled over its (u, v) ranges with the database's own
expression evaluator and compared, point cloud to point cloud, against
the vertices the shipped builder emits.  A chart is stored only while
`verify()` passes; a candidate that cannot be reproduced -- because the
builder integrates an ODE, relaxes a mesh, or does surgery -- is listed
in REASONS instead, with the honest why.

The comparison (see `verify`):

  * The chart is sampled on a dense grid over (u_range, v_range),
    trimming 2% off each non-periodic edge (periodic axes are sampled
    endpoint-exclusive over the full period).  Evaluation goes through
    `expr.parse` / the expression evaluator, so the strings checked are
    exactly the strings stored, in exactly the database's language.
  * A few builders normalise their output (centre + uniform scale to
    the 2 m cube, or a percentile-radius fit).  Those entries declare
    it, and the SAME normalisation is applied to the chart samples --
    one uniform scale and one translation, never a rotation, so a chart
    cannot be "fitted" onto a different surface.
  * Both one-sided distances must be small: every chart sample must lie
    near the oracle (max over samples), and 98% of oracle vertices must
    lie near the chart samples (the 2% slack covers the trimmed margins
    and welded apex points).  The tolerance is local: 2.5x the oracle
    cloud's median nearest-neighbour spacing as a floor, and 1.75x the
    distance to the 8th-nearest neighbour where the oracle grid is
    locally coarser or anisotropic.  On a uniform quad grid the 8th
    neighbour sits at sqrt(2) h, so 1.75x that is the same 2.5 h -- the
    local form only keeps a graded grid (Gabriel's horn bunches its
    rings toward the mouth) from failing a correct chart on
    sampling-density grounds.
  * A distance that fails against the vertex CLOUD is re-measured
    against the sampled MESH (the builders return faces as well) before
    it counts: where a grid is stretched hard in one direction -- the
    hyperbolic helicoid's corners reach cosh 4 -- a correct sample can
    sit mid-quad, far from every vertex yet on the surface.  A wrong
    chart is off the mesh too, so this loosens nothing; `_selftest`
    keeps a deliberately corrupted chart as a negative control to prove
    the gate still bites.

Parameter values: every free parameter of a builder is baked into the
chart as the OPERATOR'S default (the value a user gets by clicking the
menu entry), read from the generator source, and each entry's note
names what was fixed.  Where a record carries several implemented
variants (right-conoid, dupin-cyclide, tannery-pear) the chart fixes
the operator's default variant and says so.
"""

import importlib
import os
import sys

import numpy as np

from . import expr


# The shipped implementations live in math_art/, imported FLAT (the
# same way the generators import each other outside Blender).
_MATH_ART = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "math_art")


def _load(modname):
    if _MATH_ART not in sys.path:
        sys.path.insert(0, _MATH_ART)
    return importlib.import_module(modname)


# ----------------------------------------------------------------------
# Boy's surface: the Bryant-Kusner parametrization, expanded over the
# reals.  The shipped builder works on the unit disk in C with
# z = u e^{iv}:
#     w  = z^6 + sqrt5 z^3 - 1
#     g1 = -3/2 Im(z(1 - z^4)/w)
#     g2 = -3/2 Re(z(1 + z^4)/w)
#     g3 = Im((1 + z^6)/w) - 1/2
#     F  = (g1, g2, g3) / (g1^2 + g2^2 + g3^2)
# The expression language has no complex numbers, so the quotients are
# expanded with Re(A/w) = (Ar wr + Ai wi)/|w|^2 and
# Im(A/w) = (Ai wr - Ar wi)/|w|^2.  The strings are assembled from the
# shared fragments below rather than written out four times.
_BOY_WR = "(u**6*cos(6*v) + sqrt(5)*u**3*cos(3*v) - 1)"
_BOY_WI = "(u**6*sin(6*v) + sqrt(5)*u**3*sin(3*v))"
_BOY_W2 = "(%s**2 + %s**2)" % (_BOY_WR, _BOY_WI)
_BOY_G1 = ("(-1.5*((u*sin(v) - u**5*sin(5*v))*%s"
           " - (u*cos(v) - u**5*cos(5*v))*%s)/%s)"
           % (_BOY_WR, _BOY_WI, _BOY_W2))
_BOY_G2 = ("(-1.5*((u*cos(v) + u**5*cos(5*v))*%s"
           " + (u*sin(v) + u**5*sin(5*v))*%s)/%s)"
           % (_BOY_WR, _BOY_WI, _BOY_W2))
_BOY_G3 = ("((u**6*sin(6*v)*%s - (1 + u**6*cos(6*v))*%s)/%s - 0.5)"
           % (_BOY_WR, _BOY_WI, _BOY_W2))
_BOY_S = "(%s**2 + %s**2 + %s**2)" % (_BOY_G1, _BOY_G2, _BOY_G3)

# Surface of constant slope (Monge's sandpile), operator defaults:
# base curve r = 1 + 0.6 cos(3u), rulings along the base curve's
# outward normal raised at 45 degrees (wallis_a = 1 -> slope tan 45).
_SLOPE_TX = "(-1.8*sin(3*u)*cos(u) - (1 + 0.6*cos(3*u))*sin(u))"
_SLOPE_TY = "(-1.8*sin(3*u)*sin(u) + (1 + 0.6*cos(3*u))*cos(u))"
_SLOPE_LN = "sqrt(%s**2 + %s**2)" % (_SLOPE_TX, _SLOPE_TY)

# Hopf tori (hopf_fibration_generator).  A curve gamma on S^2 with
# colatitude beta(u) and longitude u lifts to the fibre torus
#     X = (cos(b/2) cos(u+v), cos(b/2) sin(u+v),
#          sin(b/2) cos v,    sin(b/2) sin v)  in S^3,
# stereographically projected from (0,0,0,1):  (X1,X2,X3)/(1 - X4).
# Two records use it, differing only in beta(u); both are normalised by
# the builder (bbox centre, 97th-percentile radius -> 1), replicated by
# the "hopf97" fit.
_BP_AL = "(pi/4 + (17.5*pi/180)*sin(3*u))"       # Bianchi-Pinkall alpha
_WAVY_AL = "(pi/4 + (17.5*pi/180)*cos(3*u))"     # WAVY beta/2

# Dupin ring cyclide: the shipped builder INVERTS the ring torus
# (R = 1, r = 0.45) in the sphere of radius 1.6 centred at (1.9, 0, 0);
# the chart is that inversion written out.
_CYC_D2 = ("(((1 + 0.45*cos(v))*cos(u) - 1.9)**2"
           " + ((1 + 0.45*cos(v))*sin(u))**2 + (0.45*sin(v))**2)")

# Supershape (Gielis superformula), operator default preset STARFISH:
# both superformulas are (m, n1, n2, n3, a, b) = (5, 2, 7, 7, 1, 1).
_SS_R1 = "(abs(cos(5*u/4))**7 + abs(sin(5*u/4))**7)**(-0.5)"
_SS_R2 = "(abs(cos(5*v/4))**7 + abs(sin(5*v/4))**7)**(-0.5)"


CHARTS = {
    # -- swept surfaces (math_art.helical_surface_generator) -----------
    "hyperbolic-helicoid": {
        "x": "sinh(v)*cos(2.5*u)/(1 + cosh(u)*cosh(v))",
        "y": "sinh(v)*sin(2.5*u)/(1 + cosh(u)*cosh(v))",
        "z": "cosh(v)*sinh(u)/(1 + cosh(u)*cosh(v))",
        "u_range": ("-4", "4"), "v_range": ("-4", "4"),
        "note": "torsion tau = 2.5 and extent 4 fixed to the operator "
                "defaults",
        "oracle": {"module": "helical_surface_generator",
                   "make": lambda m: m.build_hyperbolic_helicoid(
                       2.5, 4.0, 96),
                   # finite everywhere (the denominator is >= 2), so
                   # the chart can sample its exact closed domain
                   "trim": 0.0},
    },
    "seashell": {
        "x": "((1 - v/(2*pi))*(1 + cos(u)) + 0.37)*cos(2*v)",
        "y": "((1 - v/(2*pi))*(1 + cos(u)) + 0.37)*sin(2*v)",
        "z": "4*v/(2*pi) + sin(u)*(1 - v/(2*pi))",
        "u_range": ("0", "2*pi"), "v_range": ("0", "2*pi"),
        "periodic_u": True,
        "note": "whorls 2, tube aspect 1, height 4 and mouth opening "
                "0.37 fixed to the operator defaults; v = 2*pi is the "
                "apex where the tube collapses to a point",
        "oracle": {"module": "helical_surface_generator",
                   "make": lambda m: m.build_seashell(
                       2, 1.0, 4.0, 0.37, 96)},
    },
    "corkscrew": {
        "x": "0.5*cos(u)*cos(v)",
        "y": "0.5*sin(u)*cos(v)",
        "z": "0.5*sin(v) + 0.3*u",
        "u_range": ("0", "2*pi"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
        "note": "sphere radius 0.5 and twist rise 0.3 per radian fixed "
                "to the operator defaults; u is open (the twist offsets "
                "its ends in z)",
        "oracle": {"module": "helical_surface_generator",
                   "make": lambda m: m.build_corkscrew(0.5, 0.3, 96)},
    },
    "egg-box": {
        "x": "u", "y": "v", "z": "sin(u) + sin(v)",
        "u_range": ("-4", "4"), "v_range": ("-4", "4"),
        "note": "amplitude 1, wavelength 1 and extent 4 fixed to the "
                "operator defaults",
        "oracle": {"module": "helical_surface_generator",
                   "make": lambda m: m.build_translation_surface(
                       1.0, 1.0, 4.0, 96)},
    },
    "sine-torus": {
        "x": "(1 + 0.45*cos(v))*cos(u)",
        "y": "(1 + 0.45*cos(v))*sin(u)",
        "z": "0.45*sin(v)*cos(0.5*u)",
        "u_range": ("0", "2*pi"), "v_range": ("0", "2*pi"),
        "periodic_u": True, "periodic_v": True,
        "note": "major radius 1, tube 0.45 and modulation k = 1/2 (the "
                "Klein-bottle member) fixed to the operator defaults. "
                "The u seam closes onto itself with v reversed, "
                "F(u+2pi, v) = F(u, -v) -- the Klein identification",
        "oracle": {"module": "helical_surface_generator",
                   "make": lambda m: m.build_sine_torus(
                       1.0, 0.45, 0.5, 192, 64)},
    },
    "helico-conical-surface": {
        "x": "v*u*cos(v)", "y": "v*u*sin(v)", "z": "4*v*(1 - u**2)",
        "u_range": ("-1", "1"), "v_range": ("0.05", "2*pi"),
        "note": "parabolic generatrix (profile power 2), one turn and "
                "height 4 fixed to the operator defaults; the curve is "
                "rotated AND scaled by v at once, so each u-line is a "
                "conical spiral",
        "oracle": {"module": "helical_surface_generator",
                   "make": lambda m: m.build_helicone(
                       2, 1.0, 4.0, 64, 160),
                   # the ranges are exactly the builder's own safe
                   # window (v already starts at 0.05); no trim needed
                   "trim": 0.0},
    },
    "darboux-surface": {
        # A(v) C(u) + T(v) with A = Rz(v) Rx(0.6 sin 3v), C a circle of
        # radius 0.45 in the xz-plane, T a unit circle in the xy-plane.
        "x": "cos(v)*0.45*cos(u)"
             " + sin(v)*sin(0.6*sin(3*v))*0.45*sin(u) + cos(v)",
        "y": "sin(v)*0.45*cos(u)"
             " - cos(v)*sin(0.6*sin(3*v))*0.45*sin(u) + sin(v)",
        "z": "cos(0.6*sin(3*v))*0.45*sin(u)",
        "u_range": ("0", "2*pi"), "v_range": ("0", "2*pi"),
        "periodic_u": True, "periodic_v": True,
        "note": "general Darboux motion at the operator defaults: "
                "circle generatrix of size 0.45, path radius 1, tumble "
                "0.6 with 3 tumbles, one closed turn.  The builder fits "
                "its output to the 2 m cube, and verification applies "
                "the same centre-and-scale to the chart",
        "oracle": {"module": "surfaces.encyclopedia",
                   "make": lambda m: m.build_darboux(
                       res_u=96, res_v=192),
                   "fit": "fit2m"},
    },

    # -- topology menagerie (math_art.minsurf.topology) ----------------
    "klein-bottle": {
        "x": "(-2.0/15.0)*cos(u)*(3*cos(v) - 30*sin(u)"
             " + 90*cos(u)**4*sin(u) - 60*cos(u)**6*sin(u)"
             " + 5*cos(u)*cos(v)*sin(u))",
        "y": "(-1.0/15.0)*sin(u)*(3*cos(v) - 3*cos(u)**2*cos(v)"
             " - 48*cos(u)**4*cos(v) + 48*cos(u)**6*cos(v)"
             " - 60*sin(u) + 5*cos(u)*cos(v)*sin(u)"
             " - 5*cos(u)**3*cos(v)*sin(u)"
             " - 80*cos(u)**5*cos(v)*sin(u)"
             " + 80*cos(u)**7*cos(v)*sin(u))",
        "z": "(2.0/15.0)*sin(v)*(3 + 5*cos(u)*sin(u))",
        "u_range": ("0", "pi"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
        "note": "the standard bottle-shaped closed-form immersion; the "
                "u = pi rim meets the u = 0 rim under v -> pi - v",
        "oracle": {"module": "minsurf.topology",
                   "make": lambda m: m.build_klein_bottle(96, 64)},
    },
    "klein-bottle-figure-eight": {
        "x": "(2 + cos(u/2)*sin(v) - sin(u/2)*sin(2*v))*cos(u)",
        "y": "(2 + cos(u/2)*sin(v) - sin(u/2)*sin(2*v))*sin(u)",
        "z": "sin(u/2)*sin(v) + cos(u/2)*sin(2*v)",
        "u_range": ("0", "2*pi"), "v_range": ("0", "2*pi"),
        "periodic_u": True, "periodic_v": True,
        "note": "ring radius 2 fixed to the builder default.  The u "
                "seam closes with v reversed (the figure-8 makes a "
                "half-turn per revolution)",
        "oracle": {"module": "minsurf.topology",
                   "make": lambda m: m.build_klein_figure8(96, 64)},
    },
    "sudanese-mobius-band": {
        # Lawson's band in S^3, stereographically projected from the
        # farthest point p = (-1, 0, -1, 0)/sqrt2.
        "x": "sin(u)*cos(v)/(1 + (cos(u)*cos(v) + cos(2*u)*sin(v))"
             "/sqrt(2))",
        "y": "sin(2*u)*sin(v)/(1 + (cos(u)*cos(v) + cos(2*u)*sin(v))"
             "/sqrt(2))",
        "z": "(cos(u)*cos(v) - cos(2*u)*sin(v))"
             "/(sqrt(2)*(1 + (cos(u)*cos(v) + cos(2*u)*sin(v))"
             "/sqrt(2)))",
        "u_range": ("0", "pi"), "v_range": ("0", "pi"),
        "note": "u runs half way round Lawson's Klein bottle (the "
                "embedded Mobius half), v across it; the boundary "
                "v = 0 union v = pi is the round great circle",
        "oracle": {"module": "minsurf.topology",
                   "make": lambda m: m.build_sudanese_mobius(96, 48)},
    },
    "cross-cap": {
        "x": "0.5*sin(u)*sin(2*v)",
        "y": "0.5*sin(2*u)*cos(v)**2",
        "z": "0.5*cos(2*u)*cos(v)**2",
        "u_range": ("0", "2*pi"), "v_range": ("0", "pi/2"),
        "periodic_u": True,
        "note": "the standard cross-cap immersion of RP^2; antipodes "
                "(u + pi, -v) map to the same point, so the hemisphere "
                "v >= 0 covers the surface once",
        "oracle": {"module": "minsurf.topology",
                   "make": lambda m: m.build_crosscap(96, 48)},
    },
    "roman-surface": {
        "x": "sin(u)*cos(v)*sin(v)",
        "y": "cos(u)*cos(v)*sin(v)",
        "z": "sin(u)*cos(u)*cos(v)**2",
        "u_range": ("0", "2*pi"), "v_range": ("0", "pi/2"),
        "periodic_u": True,
        "note": "Steiner's Roman surface: the unit sphere pushed "
                "through (x, y, z) -> (yz, zx, xy)",
        "oracle": {"module": "minsurf.topology",
                   "make": lambda m: m.build_roman(96, 48)},
    },
    "steiner-surface": {
        # The record is the whole family of Veronese shadows; the
        # operator's default projection angle is 0, and the angle-0
        # shadow IS the Roman surface (an exact identity in the shipped
        # code, checked by its own self-test), so the chart coincides
        # with the roman-surface chart.  Stored anyway: the record is
        # distinct and its default build is this surface.
        "x": "sin(u)*cos(v)*sin(v)",
        "y": "cos(u)*cos(v)*sin(v)",
        "z": "sin(u)*cos(u)*cos(v)**2",
        "u_range": ("0", "2*pi"), "v_range": ("0", "pi/2"),
        "periodic_u": True,
        "note": "projection angle fixed to the operator default 0 "
                "degrees, where the Veronese shadow is exactly the "
                "Roman surface; other angles sweep to the cross-cap",
        "oracle": {"module": "minsurf.topology",
                   "make": lambda m: m.build_steiner(96, 48, 0.0)},
    },
    "morin-surface": {
        # Apery's order-2 member: x + iy = K(A e^{iv} + B e^{-iv}),
        # z = K cos u, with A = 2 cos u, B = sqrt2 sin u,
        # K = cos u / (sqrt2 - sin 2u sin 2v).
        "x": "cos(u)*(2*cos(u) + sqrt(2)*sin(u))*cos(v)"
             "/(sqrt(2) - sin(2*u)*sin(2*v))",
        "y": "cos(u)*(2*cos(u) - sqrt(2)*sin(u))*sin(v)"
             "/(sqrt(2) - sin(2*u)*sin(2*v))",
        "z": "cos(u)**2/(sqrt(2) - sin(2*u)*sin(2*v))",
        "u_range": ("-pi/2", "pi/2"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
        "note": "Apery order n = 2 (Morin's own halfway model, an "
                "immersed sphere) and pinch k = 1 fixed to the "
                "operator defaults; u = +-pi/2 collapse to the triple "
                "point",
        "oracle": {"module": "minsurf.topology",
                   "make": lambda m: m.build_morin(64, 96, 2, 1.0)},
    },
    "boys-surface": {
        "x": "%s/%s" % (_BOY_G1, _BOY_S),
        "y": "%s/%s" % (_BOY_G2, _BOY_S),
        "z": "%s/%s" % (_BOY_G3, _BOY_S),
        "u_range": ("0.02", "1"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
        "note": "Bryant-Kusner parametrization on the unit disk "
                "z = u e^{iv}, expanded over the reals (the language "
                "has no complex numbers).  The three denominator zeros "
                "inside the disk are the planar ends of the underlying "
                "minimal surface; they invert to the triple point, and "
                "the u grid never lands on them exactly",
        "oracle": {"module": "minsurf.topology",
                   "make": lambda m: m.build_boy(96, 48),
                   "grid": (110, 110)},
    },

    # -- ruled surfaces (math_art.ruled_surface_generator) -------------
    "hyperboloid-one-sheet": {
        "x": "(1 - v)*cos(u) + v*cos(u + 2*pi/3)",
        "y": "(1 - v)*sin(u) + v*sin(u + 2*pi/3)",
        "z": "2*v - 1",
        "u_range": ("0", "2*pi"), "v_range": ("0", "1"),
        "periodic_u": True,
        "note": "the stick hyperboloid: straight rulings between two "
                "unit circles at z = -1 and z = +1, the top rotated by "
                "the operator's default 120-degree twist (waist radius "
                "cos 60 = 1/2)",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_hyperboloid(
                       1.0, 1.0, 120.0, 96, 24)},
    },
    "hyperbolic-paraboloid": {
        "x": "u", "y": "v", "z": "u**2 - v**2",
        "u_range": ("-1", "1"), "v_range": ("-1", "1"),
        "note": "z = c((x/a)^2 - (y/b)^2) with a = b = c = 1 and "
                "extent 1, the operator defaults",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_hypar(
                       1.0, 1.0, 1.0, 1.0, 64)},
    },
    "compound-helical-cone": {
        "x": "(1 - v)*(1 + 0.18*(1 - v)*cos(6*u + 4*pi*v))*cos(u)",
        "y": "(1 - v)*(1 + 0.18*(1 - v)*cos(6*u + 4*pi*v))*sin(u)",
        "z": "2.5*v",
        "u_range": ("0", "2*pi"), "v_range": ("0", "1"),
        "periodic_u": True,
        "note": "base radius 1, height 2.5, 6 flutes of depth 0.18 "
                "winding 2 turns, straight taper, no planetary orbit "
                "-- the operator defaults.  The flutes fade linearly "
                "to arrises at the apex v = 1",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_helical_cone(
                       1.0, 2.5, 6, 0.18, 2.0, 1.0, 0.0, 1.0,
                       96, 64)},
    },
    "spiral-ruled": {
        "x": "exp(0.15*u)*(cos(u) + v*(0.15*cos(u) - sin(u)))",
        "y": "exp(0.15*u)*(sin(u) + v*(0.15*sin(u) + cos(u)))",
        "z": "v",
        "u_range": ("0", "4*pi"), "v_range": ("-1", "1"),
        "note": "Farris' spiral ruled surface at the operator "
                "defaults: log-spiral base of tightness 0.15 (plain "
                "spiral, no rosette), 2 turns, ruling slope 1 and "
                "half-length 1",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_spiral_ruled(
                       0.15, 1.0, 2.0, 1, 0.0, 1.0, 160, 20)},
    },
    "right-conoid": {
        "x": "v*cos(u)", "y": "v*sin(u)", "z": "0.5*sin(2*u)",
        "u_range": ("0", "2*pi"), "v_range": ("-1", "1"),
        "periodic_u": True,
        "note": "the record's default variant, Plucker's conoid "
                "(cylindroid) h = 0.5 sin 2u with ruling half-length "
                "1; the n-fold, Wallis, Zindler and Whitney variants "
                "share the operator but not this chart",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_conoid(
                       'PLUCKER', 0.5, 3, 1.0, 0.6, 1.0, 96, 20)},
    },
    "tangent-developable": {
        "x": "cos(u) - v*sin(u)",
        "y": "sin(u) + v*cos(u)",
        "z": "0.4*(u + v)",
        "u_range": ("0", "4*pi"), "v_range": ("0.04", "1"),
        "note": "tangent developable of the unit helix of pitch 0.4 "
                "over 2 turns (operator defaults); v starts at the "
                "0.04 edge gap off the cuspidal edge of regression "
                "(the helix itself)",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_tangent_developable(
                       1.0, 0.4, 2.0, 1.0, 0.04, 160, 16)},
    },
    "twisted-strip": {
        "x": "(1 + v*cos(u/2))*cos(u)",
        "y": "(1 + v*cos(u/2))*sin(u)",
        "z": "v*sin(u/2)",
        "u_range": ("0", "2*pi"), "v_range": ("-0.4", "0.4"),
        "periodic_u": True,
        "note": "the ruled Mobius band (1 half-twist, radius 1, "
                "half-width 0.4 -- the ruled operator's defaults).  "
                "The record's other construction is the solid strip "
                "with rectangular cross-section; this chart is the "
                "band's mid-surface.  The u seam closes with v "
                "reversed (the half-twist)",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_twist_strip(
                       1.0, 0.4, 1, 100, 8)},
    },
    "gaudi-surface": {
        "x": "2*v - 1", "y": "u", "z": "0.5*(2*v - 1)*sin(u)",
        "u_range": ("-pi", "pi"), "v_range": ("0", "1"),
        "note": "Gaudi's sinusoidal conoid z = k x sin(y/a) with "
                "k = 0.5 and a = 1 (operator defaults), as the ruling "
                "between its two boundary sinusoids at x = -1 and "
                "x = +1",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_named_ruled(
                       'GAUDI', 0.5, 1.0, 3, 1.0, 0.6, 96, 16)},
    },
    "guimard-surface": {
        "x": "cos(u)", "y": "v*sin(u)", "z": "0.5*v*sin(u)**2",
        "u_range": ("0", "2*pi"), "v_range": ("0", "1"),
        "periodic_u": True,
        "note": "the segment from M = (cos u, 0, 0) on a line to "
                "N = (cos u, sin u, 0.5 sin^2 u) on a circle "
                "(operator defaults a = b = 1, c = 0.5); both points "
                "share their x, so the rulings run in planes x = const",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_named_ruled(
                       'GUIMARD', 0.5, 1.0, 3, 1.0, 0.6, 96, 16)},
    },
    "milk-carton-surface": {
        "x": "v*cos(u)", "y": "(1 - v)*sin(u)", "z": "2*v - 1",
        "u_range": ("0", "2*pi"), "v_range": ("0", "1"),
        "periodic_u": True,
        "note": "the berlingot at the operator defaults (k = 0.5, "
                "half-height 1): horizontal sections are ellipses "
                "whose axes trade places with height, degenerating to "
                "two perpendicular segments at v = 0 and v = 1",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_named_ruled(
                       'MILK_CARTON', 0.5, 1.0, 3, 1.0, 0.6,
                       96, 16)},
    },
    "skew-ruled-cubic": {
        # The builder joins the unit circle taken rationally,
        # ((1-s^2)/(1+s^2), 2s/(1+s^2), 0) with s = tan u, to (0, 0, s)
        # on Oz; the rational point IS (cos 2u, sin 2u, 0).
        "x": "(1 - v)*cos(2*u)",
        "y": "(1 - v)*sin(2*u)",
        "z": "v*tan(u)",
        "u_range": ("-1.45", "1.45"), "v_range": ("0", "1"),
        "note": "a conic (the unit circle), the line Oz, and the "
                "identity homography s = tan u between them; the "
                "joining segments sweep the cubic.  u stays inside "
                "(-pi/2, pi/2) as in the shipped builder",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_named_ruled(
                       'RULED_CUBIC', 0.5, 1.0, 3, 1.0, 0.6,
                       96, 16)},
    },
    "surface-of-constant-slope": {
        "x": "(1 + 0.6*cos(3*u))*cos(u) + v*%s/%s"
             % (_SLOPE_TY, _SLOPE_LN),
        "y": "(1 + 0.6*cos(3*u))*sin(u) - v*%s/%s"
             % (_SLOPE_TX, _SLOPE_LN),
        "z": "v",
        "u_range": ("0", "2*pi"), "v_range": ("0", "1"),
        "periodic_u": True,
        "note": "Monge's sandpile at the operator defaults: rise from "
                "the 3-lobed base curve r = 1 + 0.6 cos 3u along its "
                "outward normal at 45 degrees (slope 1), ruling "
                "length 1",
        "oracle": {"module": "ruled_surface_generator",
                   "make": lambda m: m.build_named_ruled(
                       'CONSTANT_SLOPE', 0.5, 1.0, 3, 1.0, 0.6,
                       96, 16)},
    },

    # -- curiosities (math_art.curiosity_surface_generator) ------------
    "fresnel-elasticity-surface": {
        "x": "sqrt(sin(u)**2*cos(v)**2 + 2.25*sin(u)**2*sin(v)**2"
             " + 4*cos(u)**2)*sin(u)*cos(v)",
        "y": "sqrt(sin(u)**2*cos(v)**2 + 2.25*sin(u)**2*sin(v)**2"
             " + 4*cos(u)**2)*sin(u)*sin(v)",
        "z": "sqrt(sin(u)**2*cos(v)**2 + 2.25*sin(u)**2*sin(v)**2"
             " + 4*cos(u)**2)*cos(u)",
        "u_range": ("0", "pi"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
        "note": "the radial quartic r = sqrt(a^2 l^2 + b^2 m^2 + "
                "c^2 n^2) over unit directions, with semi-axes "
                "(1, 1.5, 2) fixed to the operator defaults",
        "oracle": {"module": "curiosity_surface_generator",
                   "make": lambda m: m.build_fresnel(
                       1.0, 1.5, 2.0, 96, 48)},
    },
    "trihyperboloid": {
        "x": "sin(u)*cos(v)/sqrt(max(max("
             "sin(u)**2 - cos(u)**2, "
             "sin(u)**2*sin(v)**2 + cos(u)**2 - sin(u)**2*cos(v)**2), "
             "cos(u)**2 + sin(u)**2*cos(v)**2 - sin(u)**2*sin(v)**2))",
        "y": "sin(u)*sin(v)/sqrt(max(max("
             "sin(u)**2 - cos(u)**2, "
             "sin(u)**2*sin(v)**2 + cos(u)**2 - sin(u)**2*cos(v)**2), "
             "cos(u)**2 + sin(u)**2*cos(v)**2 - sin(u)**2*sin(v)**2))",
        "z": "cos(u)/sqrt(max(max("
             "sin(u)**2 - cos(u)**2, "
             "sin(u)**2*sin(v)**2 + cos(u)**2 - sin(u)**2*cos(v)**2), "
             "cos(u)**2 + sin(u)**2*cos(v)**2 - sin(u)**2*sin(v)**2))",
        "u_range": ("0", "pi"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
        "note": "the radial graph r = 1/sqrt(max of the three "
                "hyperboloid forms) along the unit direction "
                "(sin u cos v, sin u sin v, cos u); the max of the "
                "three is >= 1/3, so 1 <= r <= sqrt 3 everywhere",
        "oracle": {"module": "curiosity_surface_generator",
                   "make": lambda m: m.build_trihyperboloid(96, 48)},
    },
    "paper-bag-surface": {
        "x": "v*cos(u)", "y": "(v - 1.26*u)*sin(u)", "z": "2.47*v**2",
        "u_range": ("0", "2*pi"), "v_range": ("0", "2"),
        "periodic_u": True,
        "note": "the classic constants a = 2.47 (height) and "
                "b = -1.26 (crimp) and depth 2 fixed to the operator "
                "defaults; the u = 0 and u = 2*pi boundary curves "
                "coincide, so u wraps",
        "oracle": {"module": "curiosity_surface_generator",
                   "make": lambda m: m.build_paper_bag(
                       2.47, -1.26, 96, 48, 2.0)},
    },
    "bohemian-dome": {
        "x": "cos(u)",
        "y": "1.5*sin(u) + 0.7*cos(v)",
        "z": "0.7*sin(v)",
        "u_range": ("0", "2*pi"), "v_range": ("0", "2*pi"),
        "periodic_u": True, "periodic_v": True,
        "note": "a circle of radius 0.7 swept along the (1, 1.5) "
                "ellipse in a perpendicular plane; the two radii and "
                "the swept-circle radius are the operator defaults",
        "oracle": {"module": "curiosity_surface_generator",
                   "make": lambda m: m.build_bohemian_dome(
                       1.0, 1.5, 0.7, 96, 48)},
    },
    "astroidal-ellipsoid": {
        "x": "(cos(u)*cos(v))**3",
        "y": "1.5*(sin(u)*cos(v))**3",
        "z": "2*sin(v)**3",
        "u_range": ("0", "2*pi"), "v_range": ("-pi/2", "pi/2"),
        "periodic_u": True,
        "note": "semi-axes (1, 1.5, 2) fixed to the operator "
                "defaults; equivalently the implicit "
                "(x/a)^(2/3) + (y/b)^(2/3) + (z/c)^(2/3) = 1",
        "oracle": {"module": "curiosity_surface_generator",
                   "make": lambda m: m.build_astroidal_ellipsoid(
                       1.0, 1.5, 2.0, 96, 48)},
    },
    "gabriels-horn": {
        "x": "u", "y": "cos(v)/u", "z": "sin(v)/u",
        "u_range": ("1", "6"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
        "note": "y = 1/x revolved about the x-axis, cut at the "
                "operator's default length 6; finite volume, "
                "divergent lateral area",
        "oracle": {"module": "curiosity_surface_generator",
                   "make": lambda m: m.build_gabriels_horn(
                       6.0, 64, 240),
                   # the builder bunches half its rings hard against
                   # the x = 1 mouth (t^2 grading), and nothing is
                   # singular at either end -- sample untrimmed so the
                   # mouth is covered
                   "trim": 0.0},
    },
    "neiloid": {
        "x": "sqrt(u**3)*cos(v)",
        "y": "sqrt(u**3)*sin(v)",
        "z": "u",
        "u_range": ("0.2", "1"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
        "note": "rho^2 = z^3 (a = 1), the revolution of Neile's "
                "semicubical parabola, between the operator's default "
                "base cut z = 0.2 and z = 1; the cusp tip at z = 0 "
                "stays outside the range",
        "oracle": {"module": "curiosity_surface_generator",
                   "make": lambda m: m._revolve(m._thin(
                       m.neiloid_profile(1.0, 0.2, 1.0, 384), 96),
                       96)},
    },
    "dupin-cyclide": {
        "x": "1.9 + ((1 + 0.45*cos(v))*cos(u) - 1.9)*2.56/%s" % _CYC_D2,
        "y": "(1 + 0.45*cos(v))*sin(u)*2.56/%s" % _CYC_D2,
        "z": "0.45*sin(v)*2.56/%s" % _CYC_D2,
        "u_range": ("0", "2*pi"), "v_range": ("0", "2*pi"),
        "periodic_u": True, "periodic_v": True,
        "note": "the record's default variant, the RING cyclide: the "
                "ring torus R = 1, r = 0.45 inverted in the sphere of "
                "radius 1.6 about (1.9, 0, 0) -- all operator "
                "defaults.  Inversion carries the torus's circular "
                "curvature lines to circles, which is the property "
                "that defines a Dupin cyclide",
        "oracle": {"module": "curiosity_surface_generator",
                   "make": lambda m: m.build_cyclide(
                       'RING', 1.0, 0.45, 1.9, 1.6, 96, 48)},
    },
    "tannery-pear": {
        "x": "sin(u)*cos(u)*cos(v)",
        "y": "sin(u)*cos(u)*sin(v)",
        "z": "sqrt(8)*sin(u)",
        "u_range": ("0", "pi/2"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
        "note": "half a Gerono lemniscate (rho = sin u cos u) "
                "stretched vertically by 2 sqrt2 and revolved, at the "
                "operator's default scale a = 1 -- the default "
                "TANNERY_PEAR variant (the hourglass revolves the "
                "whole lemniscate, u in [-pi/2, pi/2]).  The builder "
                "fits its output to the 2 m cube; verification "
                "applies the same centre-and-scale to the chart",
        "oracle": {"module": "surfaces.encyclopedia",
                   "make": lambda m: m.build_zoll(
                       'TANNERY_PEAR', 1.0, 96, 96),
                   "fit": "fit2m"},
    },

    # -- supershape (math_art.supershape_generator) --------------------
    "supershape": {
        "x": "%s*cos(u)*%s*cos(v)" % (_SS_R1, _SS_R2),
        "y": "%s*sin(u)*%s*cos(v)" % (_SS_R1, _SS_R2),
        "z": "%s*sin(v)" % _SS_R2,
        "u_range": ("-pi", "pi"), "v_range": ("-pi/2", "pi/2"),
        "note": "spherical product of two Gielis superformulas at the "
                "operator's default preset STARFISH, (m, n1, n2, n3, "
                "a, b) = (5, 2, 7, 7, 1, 1) for both; the longitude "
                "seam closes because the superformula radius is even "
                "in its angle",
        "oracle": {"module": "surfaces.supershape",
                   "make": lambda m: m.build_supershape_3d(
                       (5, 2, 7, 7, 1, 1), (5, 2, 7, 7, 1, 1),
                       96, 48)},
    },

    # -- Hopf tori (math_art.hopf_fibration_generator) -----------------
    "hopf-torus": {
        "x": "cos(%s)*cos(u + v)/(1 - sin(%s)*sin(v))"
             % (_WAVY_AL, _WAVY_AL),
        "y": "cos(%s)*sin(u + v)/(1 - sin(%s)*sin(v))"
             % (_WAVY_AL, _WAVY_AL),
        "z": "sin(%s)*cos(v)/(1 - sin(%s)*sin(v))"
             % (_WAVY_AL, _WAVY_AL),
        "u_range": ("0", "2*pi"), "v_range": ("0", "2*pi"),
        "periodic_u": True, "periodic_v": True,
        "note": "the operator's default WAVY curve: colatitude "
                "beta(u) = pi/2 + (35 deg) cos 3u on S^2, lifted to "
                "its Hopf fibre torus in S^3 and stereographically "
                "projected from (0,0,0,1).  The builder centres on "
                "the bounding box and scales the 97th-percentile "
                "radius to 1; verification applies the same fit to "
                "the chart",
        "oracle": {"module": "hopf_fibration_generator",
                   "make": lambda m: m.build_hopf_torus(
                       m.gamma_curve('WAVY', 240, 90.0, 3, 35.0, 0.0),
                       64),
                   "fit": "hopf97"},
    },
    "bianchi-pinkall-flat-torus": {
        "x": "cos(%s)*cos(u + v)/(1 - sin(%s)*sin(v))"
             % (_BP_AL, _BP_AL),
        "y": "cos(%s)*sin(u + v)/(1 - sin(%s)*sin(v))"
             % (_BP_AL, _BP_AL),
        "z": "sin(%s)*cos(v)/(1 - sin(%s)*sin(v))"
             % (_BP_AL, _BP_AL),
        "u_range": ("0", "2*pi"), "v_range": ("0", "2*pi"),
        "periodic_u": True, "periodic_v": True,
        "note": "the 3D-XplorMath Bianchi-Pinkall profile at the "
                "operator defaults: alpha(u) = pi/4 + (17.5 deg) "
                "sin 3u, colatitude beta = 2 alpha, lifted along the "
                "Hopf fibres and stereographically projected.  Flat "
                "in the induced metric because Hopf tori over curves "
                "of constant enclosed area are; same builder "
                "normalisation (and fit) as hopf-torus",
        "oracle": {"module": "hopf_fibration_generator",
                   "make": lambda m: m.build_hopf_torus(
                       m.gamma_curve('BIANCHI_PINKALL', 240, 90.0, 3,
                                     35.0, 0.0),
                       64),
                   "fit": "hopf97"},
    },
}


# ----------------------------------------------------------------------
# Candidates examined and NOT charted, and why.  Every reason is about
# the shipped construction itself, read from the generator source --
# not a guess about the mathematics.
REASONS = {
    # quadratures and ODE profiles -- outside the expression language
    "spherical-helicoid":
        "the meridian is a geodesic found by two quadratures (the "
        "builder cumulatively integrates theta'(s) and z'(s) with the "
        "trapezoid rule); neither integral is elementary, so no "
        "closed-form chart exists to store",
    "zoll-surface":
        "the height z(u) is a per-point Gauss-Legendre quadrature of "
        "a non-elementary integral (the Tannery variants of the same "
        "builder ARE closed-form and are charted)",
    "bouguer-dome":
        "the profile height is f(x) = a * integral of sinh(X^2/2a^2), "
        "not elementary; the builder integrates it by Simpson's rule",
    "hanging-drop-of-water":
        "the profile is the Young-Laplace equation with hydrostatic "
        "head, integrated by RK4 from the apex; no closed form",
    "k-positive-revolution":
        "the spindle/bulge meridian height is an incomplete elliptic "
        "integral of the second kind (numerical quadrature in the "
        "builder); the one elementary member, a = 1, is the sphere, "
        "already charted in charts.py",
    "minding-surface":
        "the meridian height z = integral sqrt(1 - a^2 sinh^2 t) dt "
        "is an incomplete elliptic integral, integrated numerically "
        "(Simpson) by the builder",
    "amsler-surface":
        "defined through a Painleve III ODE that the builder "
        "integrates with RK4; no closed form exists",
    "bryant-surface":
        "the CMC-1 surface is built by numerically integrating "
        "Bryant's holomorphic representation in H^3 and then mapping "
        "through the Poincare-ball model; there is no closed real "
        "chart to transcribe",
    "unduloid":
        "the Delaunay meridian is the roulette of an ellipse focus, "
        "a quadrature (elliptic integral) done numerically by the "
        "builder",
    "nodoid":
        "the Delaunay meridian is the roulette of a hyperbola focus, "
        "the same non-elementary quadrature as the unduloid",
    "bubbleton":
        "built by numerically dressing a Delaunay surface (and its "
        "component meridians are already quadratures); no closed form",
    "sphere-chain":
        "the degenerate Delaunay chain of tangent spheres is "
        "piecewise (one sphere per bead); no single smooth chart "
        "covers it",
    "elastic-torus-s3":
        "the generating elastica is solved numerically (elliptic "
        "functions / shooting) before the Hopf lift; outside the "
        "language",
    "wente-torus":
        "the immersion needs theta functions on the twisted torus; "
        "the builder evaluates it numerically and no elementary "
        "closed form exists",
    "capillary-surface":
        "solved variationally (energy minimisation with a contact "
        "line) by the CMC generator; no formula",

    # relaxation / optimisation -- the mesh IS the definition
    "willmore-surface":
        "produced by gradient-flow relaxation of the Willmore energy "
        "from a seed torus (make_willmore_torus, hundreds of "
        "iterations); the result has no parametrization at all",
    "plateau-span":
        "the soap film spanning a frame is computed by numerical "
        "relaxation (minimal_surface_toolkit); nothing closed-form to "
        "transcribe",
    "minimal-polyhedron":
        "faces are pierced and Laplacian-relaxed into membranes over "
        "the edge frame; the surface exists only as the relaxed mesh",
    "bubble-cluster":
        "soap-film cluster produced by numerical relaxation with "
        "Plateau angle conditions; no chart",
    "relaxed-bubble":
        "numerically relaxed film, as bubble-cluster",
    "pearce-saddle-surface":
        "each face is a numerically relaxed (soap-film) patch over a "
        "skew circuit; no closed form",
    "saddle-polyhedron":
        "bounded by numerically relaxed saddle-polygon patches (see "
        "pearce-saddle-surface); also piecewise, one patch per face",

    # combinatorial / surgical / piecewise constructions
    "genus-g-surface":
        "meshed by marching tetrahedra on an implicit field (a "
        "product of normalised circle factors plus k z^2); it is an "
        "implicit surface with no parametric chart",
    "non-orientable-genus-k":
        "built by mesh surgery -- cut k disks from a sphere and weld "
        "each boundary circle to itself antipodally; exact "
        "topologically, but there is no formula to store",
    "seifert-surface":
        "assembled by Seifert's algorithm from a braid word: disks "
        "stacked per Seifert circle joined by half-twisted bands "
        "along rotation-minimising frames; combinatorial geometry, "
        "no chart",
    "schwarz-lantern":
        "a POLYHEDRON inscribed in a cylinder (4 m n flat triangles); "
        "being non-smooth is its entire point, and no single smooth "
        "chart represents it",
    "steinmetz-solid":
        "the boundary of a cylinder intersection is piecewise -- "
        "several cylindrical lune patches meeting along curved edges; "
        "no single chart covers it",
    "constant-width-solid":
        "Meissner-style bodies are piecewise (spherical caps, "
        "torus-arc wedges, revolved circular arcs) assembled with "
        "edge patches; no single chart",
    "d-form":
        "the developable is determined by numerically rolling and "
        "joining the two ellipse boundaries; no closed-form "
        "parametrisation is known",
    "crochet-hyperbolic-surface":
        "grown by a discrete stitch-increase rule, row by row; the "
        "surface is the crochet fabric itself, not a formula",
    "scherk-collins-sculpture":
        "procedural geometry (Sequin's generator: Scherk tower "
        "segments warped into a toroid by a numeric pipeline of "
        "twists and bends); no chart",

    # special functions outside the language
    "spherical-harmonic-surface":
        "r = f(Y_l^m) needs the associated Legendre functions, which "
        "the expression language does not have",
    "atomic-orbital-surface":
        "hydrogenic orbitals need Laguerre polynomials and "
        "spherical harmonics; outside the language",
}

# Named in the candidate sweep but NOT candidates in the end: these
# records carry no implemented construction, so there is no shipped
# builder to verify a chart against, and an unverified formula is
# exactly what this module refuses to store.  One line each, for the
# disposition ledger.
_NO_BUILDER = ("the record has no implemented construction -- there is "
               "no shipped builder to verify a chart against, and this "
               "module stores no unverified formulas (the record's own "
               "construction block notes what is missing)")
for _slug in ("coil", "moebius-strip", "boys-planet", "dyck-surface",
              "etruscan-venus-surface", "ida-surface",
              "nested-klein-bottles", "clifford-torus",
              "parabolic-conoid", "sinusoidal-cone", "helicoidal-cone",
              "second-tractroid", "revolution-of-the-catenary",
              "revolution-of-the-sinusoid",
              "rotation-surface-with-proportional-curvatures",
              "solid-of-maximal-attraction"):
    REASONS[_slug] = _NO_BUILDER
del _slug


def chart_for(slug):
    return CHARTS.get(slug)


def reason_for(slug):
    return REASONS.get(slug)


# ----------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------

_GRID = (140, 140)     # default chart sample grid
_TRIM = 0.02           # fraction trimmed off each non-periodic edge


def _sample_chart(ch, grid, trim=_TRIM):
    """Evaluate the chart on its grid with the database's own
    evaluator.  Returns (points, n_skipped): a sample where the
    expression fails to evaluate (an isolated pole between grid nodes)
    is skipped and counted rather than crashing the run; more than a
    sliver of skips fails verification."""
    nu, nv = grid
    u0, u1 = (expr.evaluate(s) for s in ch["u_range"])
    v0, v1 = (expr.evaluate(s) for s in ch["v_range"])

    def axis(a0, a1, n, periodic):
        if periodic:
            return a0 + (a1 - a0) * np.arange(n) / n
        d = (a1 - a0) * trim
        return np.linspace(a0 + d, a1 - d, n)

    us = axis(u0, u1, nu, ch.get("periodic_u", False))
    vs = axis(v0, v1, nv, ch.get("periodic_v", False))
    trees = [expr.parse(ch[k]) for k in ("x", "y", "z")]
    env = dict(expr.CONSTANTS)
    pts, skipped = [], 0
    for u in us:
        env["u"] = float(u)
        for v in vs:
            env["v"] = float(v)
            try:
                pts.append([expr._eval(t, env) for t in trees])
            except expr.ExprError:
                skipped += 1
    return np.asarray(pts, dtype=float), skipped


def _apply_fit(P, fit):
    """Replicate the builder's own normalisation on the chart samples.

    Only uniform scale + translation -- never a rotation -- so a chart
    cannot be bent onto a different surface by the fit."""
    if fit == "fit2m":            # surfaces.encyclopedia._fit
        lo, hi = P.min(axis=0), P.max(axis=0)
        ext = float((hi - lo).max())
        return (P - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-12 else 1.0)
    if fit == "hopf97":           # hopf_fibration_generator.build_hopf_torus
        lo, hi = P.min(axis=0), P.max(axis=0)
        center = 0.5 * (lo + hi)
        rad = np.linalg.norm(P - center, axis=1)
        ref = float(np.percentile(rad, 97.0))
        return (P - center) / (ref if ref > 1e-9 else 1.0)
    return P


def _kdtree(P):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(P)
    except ImportError:
        return None


def _nearest(tree, A, B, k=1):
    """distance (and for k > 1 the k-th neighbour distances) from each
    row of A to the cloud B.  Brute-force fallback chunks the distance
    matrix so no scipy is required."""
    if tree is not None:
        d, i = tree.query(A, k=k)
        return d, i
    out_d = np.empty((len(A), k))
    out_i = np.empty((len(A), k), dtype=int)
    step = max(1, int(4e6 / max(len(B), 1)))
    for s in range(0, len(A), step):
        d2 = np.sum((A[s:s + step, None, :] - B[None, :, :]) ** 2, -1)
        idx = np.argsort(d2, axis=1)[:, :k]
        out_i[s:s + step] = idx
        out_d[s:s + step] = np.sqrt(np.take_along_axis(d2, idx, 1))
    if k == 1:
        return out_d[:, 0], out_i[:, 0]
    return out_d, out_i


def _weld(V, faces):
    """Collapse coincident vertices and remap the faces.

    Needed because some parametrizations cover their surface more than
    once (Plucker's conoid satisfies F(u + pi, -v) = F(u, v), so EVERY
    vertex is emitted twice) and some builders leave split seams of
    coincident duplicates; either would zero out the nearest-neighbour
    statistics."""
    key = np.round(np.asarray(V, dtype=float), 9)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return uniq, [[int(inv[i]) for i in f] for f in faces]


def _tris_and_incidence(V, faces):
    """Fan-triangulate the faces; return (tri index array (m, 3),
    per-vertex list of incident triangle ids).  Degenerate fans from
    welded poles are kept -- the distance routine treats a collapsed
    triangle as its own edge."""
    tris = []
    for f in faces:
        for i in range(1, len(f) - 1):
            tris.append((f[0], f[i], f[i + 1]))
    T = np.asarray(tris, dtype=int)
    incident = [[] for _ in range(len(V))]
    for t, (a, b, c) in enumerate(T):
        incident[a].append(t)
        incident[b].append(t)
        incident[c].append(t)
    return T, incident


def _pt_tri_dist(p, A, B, C):
    """Distance from point p to each triangle (A[i], B[i], C[i])
    (Ericson's closest-point-on-triangle, vectorised over triangles;
    robust to degenerate triangles, which clamp to edges/vertices)."""
    ab, ac, ap = B - A, C - A, p[None, :] - A
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    bp = p[None, :] - B
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    cp = p[None, :] - C
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    v = np.where(np.abs(denom) > 1e-30, vb / np.where(denom == 0, 1, denom), 0.0)
    w = np.where(np.abs(denom) > 1e-30, vc / np.where(denom == 0, 1, denom), 0.0)
    P = A + v[:, None] * ab + w[:, None] * ac       # interior candidate
    # region clamps
    # vertex A
    P = np.where(((d1 <= 0) & (d2 <= 0))[:, None], A, P)
    # vertex B
    P = np.where(((d3 >= 0) & (d4 <= d3))[:, None], B, P)
    # vertex C
    P = np.where(((d6 >= 0) & (d5 <= d6))[:, None], C, P)
    # edge AB
    t = d1 / np.where(d1 - d3 == 0, 1, d1 - d3)
    onAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    P = np.where(onAB[:, None], A + np.clip(t, 0, 1)[:, None] * ab, P)
    # edge AC
    t = d2 / np.where(d2 - d6 == 0, 1, d2 - d6)
    onAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    P = np.where(onAC[:, None], A + np.clip(t, 0, 1)[:, None] * ac, P)
    # edge BC
    t = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1,
                             (d4 - d3) + (d5 - d6))
    onBC = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    P = np.where(onBC[:, None], B + np.clip(t, 0, 1)[:, None] * (C - B), P)
    return np.sqrt(np.min(np.sum((P - p[None, :]) ** 2, axis=1)))


def _refine_to_mesh(pts, d0, tree, V, T, incident, mask):
    """For the points flagged by `mask`, replace the vertex-cloud
    distance d0 with the distance to the MESH around their nearest
    vertices.  A grid stretched hard in one direction (the hyperbolic
    helicoid's corners reach cosh 4) puts its vertices far apart along
    the stretch, so a correct chart sample can sit mid-quad, far from
    every vertex but ON the surface; measuring to the triangles is the
    honest metric there.  Only flagged points pay the cost."""
    d = d0.copy()
    where = np.nonzero(mask)[0]
    if len(where) == 0:
        return d
    _, knn = _nearest(tree, pts[where], V, k=12)
    for row, i in enumerate(where):
        cand = set()
        for j in np.atleast_1d(knn[row]):
            cand.update(incident[int(j)])
        if not cand:
            continue
        t = T[sorted(cand)]
        d[i] = min(d[i], _pt_tri_dist(pts[i], V[t[:, 0]], V[t[:, 1]],
                                      V[t[:, 2]]))
    return d


def _chart_faces(grid, ch):
    """Grid faces for the chart samples (periodic axes wrap), so the
    reverse check can measure oracle vertices against the chart MESH."""
    nu, nv = grid
    wrap_u = ch.get("periodic_u", False)
    wrap_v = ch.get("periodic_v", False)
    faces = []
    iu = nu if wrap_u else nu - 1
    jv = nv if wrap_v else nv - 1
    for i in range(iu):
        i1 = (i + 1) % nu
        for j in range(jv):
            j1 = (j + 1) % nv
            faces.append((i * nv + j, i1 * nv + j,
                          i1 * nv + j1, i * nv + j1))
    return faces


def verify(slug):
    """(ok, detail): reproduce the chart against the shipped builder.

    Point-cloud identity up to the declared normalisation: max chart ->
    oracle distance and 98th-percentile oracle -> chart distance must
    both sit inside the local sampling tolerance (see module
    docstring).  Distances that fail against the vertex cloud are
    re-measured against the sampled mesh before they count as
    failures."""
    ch = CHARTS.get(slug)
    if ch is None:
        return False, "no chart stored for %r" % slug
    spec = ch["oracle"]
    mod = _load(spec["module"])
    result = spec["make"](mod)
    O_raw = np.asarray(result[0], dtype=float)
    faces_raw = result[1]
    if O_raw.ndim != 2 or O_raw.shape[1] != 3:
        return False, "oracle returned shape %r" % (O_raw.shape,)
    O, ofaces = _weld(O_raw, faces_raw)
    grid = spec.get("grid", _GRID)
    C, skipped = _sample_chart(ch, grid, spec.get("trim", _TRIM))
    if skipped > 0.001 * (len(C) + skipped):
        return False, "%d of %d chart samples failed to evaluate" % (
            skipped, len(C) + skipped)
    C = _apply_fit(C, spec.get("fit"))

    # local sampling scale of the (welded) oracle cloud
    treeO = _kdtree(O)
    dO, _ = _nearest(treeO, O, O, k=9)
    nn = dO[:, 1]
    med = float(np.median(nn[nn > 1e-9])) if np.any(nn > 1e-9) else 0.0
    if med <= 0.0:
        return False, "oracle cloud is degenerate (zero spacing)"
    d8 = dO[:, 8]
    allow = np.maximum(2.5 * med, 1.75 * d8)

    OT, Oinc = _tris_and_incidence(O, ofaces)

    # chart -> oracle: EVERY chart sample must be near the oracle.
    # Anything above 80% of its allowance is re-measured against the
    # mesh, not just the failures, so the verdict does not ride on a
    # vertex happening to sit at 0.99 of tolerance.
    dCO, idx = _nearest(treeO, C, O)
    dCO = _refine_to_mesh(C, dCO, treeO, O, OT, Oinc,
                          dCO > 0.8 * allow[idx])
    fwd = float(np.max(dCO / allow[idx]))

    # oracle -> chart: 98% of oracle vertices must be near the chart
    sub = np.arange(0, len(O), max(1, len(O) // 6000))
    treeC = _kdtree(C)
    dOC, _ = _nearest(treeC, O[sub], C)
    bad = dOC > 0.8 * allow[sub]
    if np.any(bad) and skipped == 0:
        CT, Cinc = _tris_and_incidence(C, _chart_faces(grid, ch))
        dOC = _refine_to_mesh(O[sub], dOC, treeC, C, CT, Cinc, bad)
    frac = float(np.mean(dOC > allow[sub]))

    ok = fwd <= 1.0 and frac <= 0.02
    detail = ("fwd max %.2f of tol, %.1f%% of oracle beyond tol "
              "(tol floor 2.5 x %.4g)" % (fwd, 100.0 * frac, med))
    return ok, detail


def _selftest():
    """Every stored chart must (a) be in the expression language with
    only u and v free, and (b) reproduce its shipped builder."""
    bad = []
    both = sorted(set(CHARTS) & set(REASONS))
    assert not both, "slugs in both CHARTS and REASONS: %r" % both

    for slug, ch in sorted(CHARTS.items()):
        for key in ("x", "y", "z"):
            free = expr.free_names(ch[key])
            if not free <= {"u", "v"}:
                bad.append("%s: %s references %r" % (slug, key, free))
        for key in ("u_range", "v_range"):
            for s in ch[key]:
                if expr.free_names(s):
                    bad.append("%s: range %r is not constant" % (slug, s))
    if bad:
        raise AssertionError("charts outside the language:\n  "
                             + "\n  ".join(bad))

    assert chart_for("egg-box") is CHARTS["egg-box"]
    assert chart_for("no-such-slug") is None
    assert reason_for("zoll-surface")
    assert reason_for("egg-box") is None

    for slug in sorted(CHARTS):
        ok, detail = verify(slug)
        print("  %-28s %s  (%s)" % (slug, "ok" if ok else "FAIL", detail))
        if not ok:
            bad.append("%s: %s" % (slug, detail))
    if bad:
        raise AssertionError("charts failing reproduction against the "
                             "shipped builders:\n  " + "\n  ".join(bad))

    # negative control: the gate must REJECT a wrong chart, or the 39
    # passes above prove nothing.  The egg box with its amplitude
    # quietly scaled by 1.3 is exactly the kind of transcription slip
    # this module exists to catch.
    wrong = dict(CHARTS["egg-box"])
    wrong["z"] = "1.3*(sin(u) + sin(v))"
    CHARTS["_negative-control"] = wrong
    try:
        ok, detail = verify("_negative-control")
    finally:
        del CHARTS["_negative-control"]
    assert not ok, ("a chart with a wrong amplitude passed "
                    "verification: " + detail)
    print("  negative control: a mis-scaled egg box is rejected (%s)"
          % detail)
    print("RESULT: OK  (surfdb.parcharts, %d charts verified against "
          "the shipped builders)" % len(CHARTS))
