# Algebraic surfaces: the zero sets of polynomials in three variables.
#
# Part of the Math Art surfaces engine (`math_art/surfaces/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Cayley's nodal cubic, the Clebsch diagonal surface with its 27 real
# lines, Kummer's quartic with sixteen nodes, Barth's sextic and decic
# -- the classical surfaces, each the zero set of one polynomial and
# each famous for the singularities it achieves.  Contoured by marching
# tetrahedra, which cannot produce the ambiguous cases marching cubes
# can.
#
# References:
# - A. Cayley, "A Memoir on Cubic Surfaces", Phil. Trans. 159, 1869.
# - A. Clebsch, "Ueber die Flachen vierter Ordnung...", Journal fur die
#   reine und angewandte Mathematik 69, 1868 -- the diagonal surface.
# - W. Barth, "Two projective surfaces with many nodes admitting the
#   symmetries of the icosahedron", J. Algebraic Geometry 5, 1996.
# - H. Hauser, "Bildergalerie algebraischer Flaechen", Universitaet
#   Wien -- the named gallery surfaces in the _HAUSER block below.

import math
import numpy as np


_PHI = (1.0 + math.sqrt(5.0)) / 2.0          # golden ratio


_SQRT2 = math.sqrt(2.0)


def _toolkit():
    """The sibling `minsurf` engine package supplies marching_tets.
    Relative import when installed as part of the Math Art package,
    absolute when this file runs standalone next to the toolkit."""
    # two dots: `minsurf` is a sibling of this PACKAGE, not a member of
    # it.  A single dot resolves to `surfaces.minsurf`, which does not
    # exist -- and the flat fallback then hides the mistake anywhere
    # `math_art/` is on sys.path, which is true of the headless test
    # runner and false inside Blender.
    try:
        from .. import minsurf as mst
    except ImportError:
        import minsurf as mst
    return mst


def _f_clebsch(x, y, z, mu):
    # 81(x^3+y^3+z^3) - 189(sum of x^2 y terms) + 54xyz
    #   + 126(xy+xz+yz) - 9(x^2+y^2+z^2) - 9(x+y+z) + 1 = 0
    # Future option: overlay the famous 27 real lines of this cubic
    # as curve objects (not implemented).
    x2, y2, z2 = x * x, y * y, z * z
    return (81.0 * (x2 * x + y2 * y + z2 * z)
            - 189.0 * (x2 * y + x2 * z + y2 * x + y2 * z
                       + z2 * x + z2 * y)
            + 54.0 * x * y * z
            + 126.0 * (x * y + x * z + y * z)
            - 9.0 * (x2 + y2 + z2)
            - 9.0 * (x + y + z) + 1.0)


def _f_cayley(x, y, z, mu):
    # 4(x^2+y^2+z^2) + 16xyz - 1 = 0 -- four conical nodes, the
    # maximum possible on a cubic surface
    return 4.0 * (x * x + y * y + z * z) + 16.0 * x * y * z - 1.0


def _f_kummer(x, y, z, mu):
    # (x^2+y^2+z^2 - mu^2)^2 - lambda p q r s = 0 with the four
    # tetrahedral tangent planes p, q, r, s; 16 nodes for generic mu
    mu2 = mu * mu
    lam = (3.0 * mu2 - 1.0) / (3.0 - mu2)
    p = 1.0 - z - _SQRT2 * x
    q = 1.0 - z + _SQRT2 * x
    r = 1.0 + z + _SQRT2 * y
    s = 1.0 + z - _SQRT2 * y
    core = x * x + y * y + z * z - mu2
    return core * core - lam * p * q * r * s


def _f_barth(x, y, z, mu):
    # 4(phi^2 x^2 - y^2)(phi^2 y^2 - z^2)(phi^2 z^2 - x^2)
    #   - (1+2 phi)(x^2+y^2+z^2-1)^2 = 0, phi the golden ratio
    p2 = _PHI * _PHI
    x2, y2, z2 = x * x, y * y, z * z
    w = x2 + y2 + z2 - 1.0
    return (4.0 * (p2 * x2 - y2) * (p2 * y2 - z2) * (p2 * z2 - x2)
            - (1.0 + 2.0 * _PHI) * w * w)


def _f_togliatti(x, y, z, mu):
    # 64(x-1)(x^4 - 4x^3 - 10x^2 y^2 - 4x^2 + 16x - 20x y^2
    #   + 5y^4 + 16 - 20y^2)
    #   - 5 a (2z - a)(4(x^2+y^2-z^2) + (1+3 sqrt5))^2 = 0,
    # a = sqrt(5 - sqrt5)
    a = math.sqrt(5.0 - math.sqrt(5.0))
    x2, y2 = x * x, y * y
    quart = (x2 * x2 - 4.0 * x2 * x - 10.0 * x2 * y2 - 4.0 * x2
             + 16.0 * x - 20.0 * x * y2 + 5.0 * y2 * y2
             + 16.0 - 20.0 * y2)
    q = 4.0 * (x2 + y2 - z * z) + (1.0 + 3.0 * math.sqrt(5.0))
    return (64.0 * (x - 1.0) * quart
            - 5.0 * a * (2.0 * z - a) * q * q)


def _f_heart(x, y, z, mu):
    # Taubin's heart (z up, lobes on top):
    # (x^2 + 9/4 y^2 + z^2 - 1)^3 - x^2 z^3 - 9/80 y^2 z^3 = 0
    z3 = z * z * z
    w = x * x + 2.25 * y * y + z * z - 1.0
    return w * w * w - x * x * z3 - 0.1125 * y * y * z3


def _f_dingdong(x, y, z, mu):
    # x^2 + y^2 - (1 - z) z^2 = 0 -- a droplet sitting on a cone
    return x * x + y * y - (1.0 - z) * z * z


def _f_chmutov(x, y, z, mu):
    # T6(x) + T6(y) + T6(z) = 0 with the degree-6 Chebyshev
    # polynomial T6(t) = 32t^6 - 48t^4 + 18t^2 - 1
    def t6(t):
        t2 = t * t
        return ((32.0 * t2 - 48.0) * t2 + 18.0) * t2 - 1.0
    return t6(x) + t6(y) + t6(z)


def _f_tangle(x, y, z, mu):
    # tangle cube: x^4 - 5x^2 + y^4 - 5y^2 + z^4 - 5z^2 + 11.8 = 0
    x2, y2, z2 = x * x, y * y, z * z
    return (x2 * x2 - 5.0 * x2 + y2 * y2 - 5.0 * y2
            + z2 * z2 - 5.0 * z2 + 11.8)


def _f_monkey(x, y, z, mu, n=3):
    # n-fold monkey saddle: z = Re((x+iy)^n) = rho^n cos(n*phi), the
    # graph of the degree-n harmonic polynomial (real part of the
    # holomorphic w^n).  n = 2 is the ordinary saddle z = x^2 - y^2,
    # n = 3 the classic monkey saddle z = x^3 - 3xy^2 (two legs and a
    # tail), n >= 4 the higher-fold saddles -- forms Robert Fathauer
    # renders as ceramic saddle sheets.  Scaling: inside the unit clip
    # ball rho <= 1 the height obeys |Re w^n| <= rho^n <= 1, so the
    # unit-coefficient polynomial already sits in frame; the spherical
    # clip trims the sheet to a rim that waves up and down n times.
    return z - ((x + 1j * y) ** int(n)).real


# Each preset stores its own clip region framing the interesting part
# of the (usually unbounded) surface: 'BALL' with radius r, or 'BOX'
# with half-extent r.
PRESETS = {
    'CLEBSCH': ("Clebsch Diagonal Cubic", _f_clebsch, 'BALL', 3.0),
    'CAYLEY': ("Cayley Nodal Cubic", _f_cayley, 'BALL', 1.4),
    'KUMMER': ("Kummer Quartic", _f_kummer, 'BALL', 2.2),
    'BARTH': ("Barth Sextic", _f_barth, 'BALL', 2.0),
    'TOGLIATTI': ("Togliatti Quintic", _f_togliatti, 'BALL', 5.0),
    'HEART': ("Taubin Heart", _f_heart, 'BOX', 1.5),
    'DINGDONG': ("Ding-dong", _f_dingdong, 'BALL', 1.5),
    'CHMUTOV': ("Chmutov Sextic", _f_chmutov, 'BOX', 1.1),
    'TANGLE': ("Tangle Cube", _f_tangle, 'BOX', 2.4),
    'MONKEY': ("Monkey Saddle (n-fold)", _f_monkey, 'BALL', 1.0),
}


# ======================================================================
# Herwig Hauser's gallery of algebraic surfaces
# ======================================================================
# Sixty-three surfaces from Hauser's "Bildergalerie algebraischer
# Flaechen", each the zero set of one low-degree polynomial and each
# named for what it looks like rather than for a theorem -- Zitrus,
# Seepferdchen, Schneeflocke, Himmel und Hoelle.  Several are singular
# by construction (Kreuz is xyz = 0, three coordinate planes; Kegel and
# Diabolo are cones), so unlike the classical surfaces above they are
# NOT expected to mesh watertight, and the rim where they leave the
# clip box is a real boundary rather than a defect.
#
# Every row carries the equation AS PRINTED in the gallery caption
# alongside the lambda.  That is not decoration: `_selftest` parses the
# printed form back into an expression and checks it agrees with the
# lambda numerically.  Both are written from the same caption, so this
# does not validate the source -- it stops the two copies drifting
# apart later, which is the failure mode that would otherwise ship a
# plausible-looking wrong surface without erroring.
#
# Deliberately NOT transcribed:
#   Quaste, Croissant, Wendel, Moebius -- the gallery names these but
#       prints no equation for them (image-only cells).
#   Zweiloch -- the gallery prints xyz + yz + 2z^5 = 0 for BOTH
#       Zeppelin and Zweiloch.  One of the two is wrong at the source
#       and there is no way to tell which, so only Zeppelin ships.
#   Torus, Sphaere, Zylinder, Cylinder -- trivial; already primitives.
#   Dingdong -- already shipped above as DINGDONG.
#
# The clip extent on each row was chosen by measurement rather than by
# eye: a sweep over BOX/BALL and several radii, keeping the smallest
# that yields a solid patch whose bounding box is not a degenerate slab.
#
# References:
# - H. Hauser, "Bildergalerie algebraischer Flaechen", Universitaet
#   Wien.  https://homepage.univie.ac.at/herwig.hauser/bildergalerie/
# - S. Amethyst, 3D-printed renderings of the same gallery,
#   https://silviana.org/gallery/hauser/
#
# Row layout: (key, label, equation as printed, clip shape, clip, F)
_HAUSER = (
    ('CALYX', "Calyx",
     "x^2+y^2z^3 = z^4",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**2 + y**2 * z**3) - (z**4)),
    ('CALYPSO', "Calypso",
     "x^2+y^2z = z^2",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**2 + y**2 * z) - (z**2)),
    ('COLUMPIUS', "Columpius",
     "x^3y+xz^3+y^3z+z^3+7z^2+5z=0",
     'BOX', 1.2,
     lambda x, y, z, mu:
         x**3 * y + x * z**3 + y**3 * z + z**3 + 7 * z**2 + 5 * z),
    ('CUBE', "Cube",
     "x^6+y^6+z^6=1",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**6 + y**6 + z**6) - (1)),
    ('DATTEL', "Dattel",
     "3x^2+3y^2+z^2=1",
     'BOX', 1.2,
     lambda x, y, z, mu: (3 * x**2 + 3 * y**2 + z**2) - (1)),
    ('DAISY', "Daisy",
     "(x^2 - y^3)^2=(z^2-y^2)^3",
     'BOX', 1.2,
     lambda x, y, z, mu: ((x**2 - y**3)**2) - ((z**2 - y**2)**3)),
    ('DISTEL', "Distel",
     "x^2+y^2+z^2+1000 (x^2+y^2)(x^2+z^2)(y^2+z^2)=1",
     'BOX', 1.2,
     lambda x, y, z, mu:
         (x**2 + y**2 + z**2 + 1000 * (x**2 + y**2) * (x**2 + z**2)
         * (y**2 + z**2)) - (1)),
    ('DURCHBLICK', "Durchblick",
     "x^3y+xz^3+y^3z+z^3+5z = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**3 * y + x * z**3 + y**3 * z + z**3 + 5 * z),
    ('EISTUETE', "Eistuete",
     "(x^2+y^2)^3 = 4x^2y^2(z^2+1)",
     'BOX', 1.2,
     lambda x, y, z, mu:
         ((x**2 + y**2)**3) - (4 * x**2 * y**2 * (z**2 + 1))),
    ('EVE', "Eve",
     "5x^2 + 2xz^2 + 5y^6 + 15y^4 + 5z^2 = 15y^5 + 5y^3",
     'BOX', 1.2,
     lambda x, y, z, mu:
         (5 * x**2 + 2 * x * z**2 + 5 * y**6 + 15 * y**4 + 5 *
         z**2) - (15 * y**5 + 5 * y**3)),
    ('FLIRT', "Flirt",
     "x^2-x^3+y^2+y^4+z^3-10z^4=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 - x**3 + y**2 + y**4 + z**3 - 10 * z**4),
    ('GEISHA', "Geisha",
     "x^2yz + x^2z^2 = y^3z + y^3",
     'BOX', 1.2,
     lambda x, y, z, mu:
         (x**2 * y * z + x**2 * z**2) - (y**3 * z + y**3)),
    ('HARLEKIN', "Harlekin",
     "x^3z + 10x^2y + xy^2 + yz^2 = z^3",
     'BOX', 1.2,
     lambda x, y, z, mu:
         (x**3 * z + 10 * x**2 * y + x * y**2 + y * z**2) - (z**3)),
    ('HELIX', "Helix",
     "6x^2 - 2x^4 = y^2z^2",
     'BOX', 1.2,
     lambda x, y, z, mu: (6 * x**2 - 2 * x**4) - (y**2 * z**2)),
    ('HERZ', "Herz",
     "y^2+z^3-z^4-x^2z^2 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: y**2 + z**3 - z**4 - x**2 * z**2),
    ('HIMMEL_UND_HOELLE', "Himmel und Hoelle",
     "x^2-y^2z^2=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 - y**2 * z**2),
    ('KOLIBRI', "Kolibri",
     "x^3 + x^2z^2 - y^2",
     'BOX', 1.2,
     lambda x, y, z, mu: x**3 + x**2 * z**2 - y**2),
    ('LEOPOLD', "Leopold",
     "x^2y^2z^2+3x^2+3y^2+z^2=1",
     'BOX', 1.2,
     lambda x, y, z, mu:
         (x**2 * y**2 * z**2 + 3 * x**2 + 3 * y**2 + z**2) - (1)),
    ('OCTDONG', "Octdong",
     "x^2 + y^2 + z^4 = z^2",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**2 + y**2 + z**4) - (z**2)),
    ('PLOP', "Plop",
     "x^2 + (z+y^2)^3 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 + (z + y**2)**3),
    ('SEEPFERDCHEN', "Seepferdchen",
     "(x^2-y^3)^2=(x+y^2)z^3",
     'BOX', 1.2,
     lambda x, y, z, mu: ((x**2 - y**3)**2) - ((x + y**2) * z**3)),
    ('SOFA', "Sofa",
     "x^2+y^3+z^5 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 + y**3 + z**5),
    ('SOLITUDE', "Solitude",
     "x^2yz +xy^2+y^3+y^3z=x^2z^2",
     'BOX', 1.2,
     lambda x, y, z, mu:
         (x**2 * y * z + x * y**2 + y**3 + y**3 * z) - (x**2 *
         z**2)),
    ('SUESS', "Suess",
     "(x^2+9/4y^2+z^2-1)^3 - x^2z^3-9/80y^2z^3=0",
     'BOX', 1.2,
     lambda x, y, z, mu:
         (x**2 + 9 / 4 * y**2 + z**2 - 1)**3 - x**2 * z**3 - 9 / 80
         * y**2 * z**3),
    ('TANZ', "Tanz",
     "x^4-x^2-y^2z^2=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**4 - x**2 - y**2 * z**2),
    ('TAUBE', "Taube",
     "256z^3-128x^2z^2+16x^4z+144xy^2z - 4x^3y^2-27y^4 =0",
     'BOX', 1.2,
     lambda x, y, z, mu:
         256 * z**3 - 128 * x**2 * z**2 + 16 * x**4 * z + 144 * x *
         y**2 * z - 4 * x**3 * y**2 - 27 * y**4),
    ('SPITZ', "Spitz",
     "(y^3-x^2-z^2)^3 = 27x^2y^3z^2",
     'BOX', 1.2,
     lambda x, y, z, mu:
         ((y**3 - x**2 - z**2)**3) - (27 * x**2 * y**3 * z**2)),
    ('TOBEL', "Tobel",
     "x^3 z + x^2 + yz^3 + z^4 = 3xyz",
     'BOX', 1.2,
     lambda x, y, z, mu:
         (x**3 * z + x**2 + y * z**3 + z**4) - (3 * x * y * z)),
    ('VIS_A_VIS', "Vis a vis",
     "x^2-x^3+y^2+y^4+z^3-z^4=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 - x**3 + y**2 + y**4 + z**3 - z**4),
    ('WEDELN', "Wedeln",
     "x^3 = y (1-z^2)^2",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**3) - (y * (1 - z**2)**2)),
    ('WINDKANAL', "Windkanal",
     "- x^2 + y^4 + z^4 - xyz = 100",
     'BOX', 3.0,
     lambda x, y, z, mu: (-x**2 + y**4 + z**4 - x * y * z) - (100)),
    ('XANO', "Xano",
     "x^4 +z^3 = yz^2",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**4 + z**3) - (y * z**2)),
    ('ZITRUS', "Zitrus",
     "x^2+z^2 = y^3(y-1)^3",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**2 + z**2) - (y**3 * (y - 1)**3)),
    ('DROMEDAR', "Dromedar",
     "x^4 - 3x^2 + y^2+z^3 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**4 - 3 * x**2 + y**2 + z**3),
    ('ZEPPELIN', "Zeppelin",
     "xyz+yz+2z^5= 0",
     'BOX', 1.2,
     lambda x, y, z, mu: x * y * z + y * z + 2 * z**5),
    ('MICHELANGELO', "Michelangelo",
     "x^2+y^4+y^3z^2=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 + y**4 + y**3 * z**2),
    ('STERN', "Stern",
     "x^2y^2 + y^2z^2 + x^2z^2 + 100 ( x^2 + y^2 + z^2 - 1)^3 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu:
         x**2 * y**2 + y**2 * z**2 + x**2 * z**2 + 100 * (x**2 +
         y**2 + z**2 - 1)**3),
    ('LIMAO', "Limao",
     "x^2-y^3z^3 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 - y**3 * z**3),
    ('WHITNEY', "Whitney",
     "x^2-y^2z=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 - y**2 * z),
    ('BUGGLE', "Buggle",
     "x^4y^2+y^4x^2-x^2y^2+z^6=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**4 * y**2 + y**4 * x**2 - x**2 * y**2 + z**6),
    ('DIABOLO', "Diabolo",
     "x^2 = (y^2+ z^2)^2",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**2) - ((y**2 + z**2)**2)),
    ('DULLO', "Dullo",
     "(x^2+y^2+z^2)^2-(x^2+y^2) = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**2 + y**2 + z**2)**2 - (x**2 + y**2)),
    ('MIAU', "Miau",
     "x^2yz + x^2z^2 + 2 y^3z + 3 y^3 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu:
         x**2 * y * z + x**2 * z**2 + 2 * y**3 * z + 3 * y**3),
    ('TRICHTER', "Trichter",
     "x^2 + z^3 = y^2z^2",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**2 + z**3) - (y**2 * z**2)),
    ('NEPALI', "Nepali",
     "(xy-z^3-1)^2 + (x^2+y^2-1)^3 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: (x * y - z**3 - 1)**2 + (x**2 + y**2 - 1)**3),
    ('PILZCHEN', "Pilzchen",
     "(z^3 - 1)^2 + (x^2+y^2-1)^3=0",
     'BOX', 1.2,
     lambda x, y, z, mu: (z**3 - 1)**2 + (x**2 + y**2 - 1)**3),
    ('SUBWAY', "Subway",
     "x^2y^2 = (z^2-1)^3",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**2 * y**2) - ((z**2 - 1)**3)),
    ('POLSTERZIPF', "Polsterzipf",
     "(x^3-1)^2 + (y^3-1)^2+ (z^2-1)^3 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**3 - 1)**2 + (y**3 - 1)**2 + (z**2 - 1)**3),
    ('CRIXXI', "Crixxi",
     "(y^2+z^2-1)^2 +(x^2+y^2-1)^3 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: (y**2 + z**2 - 1)**2 + (x**2 + y**2 - 1)**3),
    ('BERG', "Berg",
     "x^2+y^2z^2+z^3 = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 + y**2 * z**2 + z**3),
    ('GUPF', "Gupf",
     "x^2+y^2+z=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 + y**2 + z),
    ('KEGEL', "Kegel",
     "x^2+y^2-z^2=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 + y**2 - z**2),
    ('WIGWAM', "Wigwam",
     "x^2+y^2z^3=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 + y**2 * z**3),
    ('TUELLE', "Tuelle",
     "yz(x^2+y-z) = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: y * z * (x**2 + y - z)),
    ('PIPE', "Pipe",
     "x^2-z=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 - z),
    ('FANFARE', "Fanfare",
     "-x^3+z^2+y^2=0",
     'BOX', 1.2,
     lambda x, y, z, mu: -x**3 + z**2 + y**2),
    ('KREUZ', "Kreuz",
     "xyz=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x * y * z),
    ('SPINDEL', "Spindel",
     "x^2+y^2-z^2= 1",
     'BOX', 1.2,
     lambda x, y, z, mu: (x**2 + y**2 - z**2) - (1)),
    ('TWILIGHT', "Twilight",
     "(z^3-2)^2 +(x^2+y^2-3)^3 =0",
     'BOX', 1.2,
     lambda x, y, z, mu: (z**3 - 2)**2 + (x**2 + y**2 - 3)**3),
    ('UFO', "Ufo",
     "z^2-x^2-y^2 = 1",
     'BOX', 1.2,
     lambda x, y, z, mu: (z**2 - x**2 - y**2) - (1)),
    ('ZECK', "Zeck",
     "x^2+y^2-z^3(1-z) = 0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 + y**2 - z**3 * (1 - z)),
    ('SATTEL', "Sattel",
     "x^2+y^2z+z^3=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**2 + y**2 * z + z**3),
    ('SCHNEEFLOCKE', "Schneeflocke",
     "x^3+y^2z^3+yz^4=0",
     'BOX', 1.2,
     lambda x, y, z, mu: x**3 + y**2 * z**3 + y * z**4),
)

for _key, _label, _eq, _shape, _clip, _fn in _HAUSER:
    PRESETS[_key] = (_label, _fn, _shape, _clip)

HAUSER_EQUATION = {k: eq for (k, _l, eq, _s, _c, _f) in _HAUSER}


# Caption-parsing patterns for _printed_expr (module level so the
# backslashes live in exactly one place).
POW = r'\^(\d+)'
NUMVAR = r'(\d)(?=[xyz(])'
VARVAR = r'([xyz)])(?=[xyz(])'
PARNUM = r'(\))(?=\d)'


def _printed_expr(eq):
    """Parse a printed gallery equation into a Python expression.

    Handles the two things the captions do that Python does not:
    superscripts (x^2) and implicit multiplication (xyz, 3x, 9/4y^2,
    ")("), and folds "lhs = rhs" into "lhs - rhs".  Only `_selftest`
    uses this -- it is the cross-check against each row's lambda, not
    a runtime path."""
    import re

    def side(s):
        s = s.replace(' ', '')
        # x^2 -> x**2
        s = re.sub(POW, lambda m: '**' + m.group(1), s)
        # 3x -> 3*x,  xy -> x*y,  )2 -> )*2
        for pat in (NUMVAR, VARVAR, PARNUM):
            s = re.sub(pat, lambda m: m.group(1) + '*', s)
        return s

    if '=' in eq:
        lhs, rhs = eq.split('=', 1)
        return '(%s) - (%s)' % (side(lhs), side(rhs))
    return side(eq)


# Which family each preset belongs to.  The operator turns this into a
# Family dropdown that filters the Surface list: an 80-entry flat enum
# is unusable, and the families are how the sources group them anyway.
FAMILIES = (
    ('CLASSICAL', "Classical"),
    ('HAUSER', "Hauser Gallery"),
)

SURFACE_FAMILY = {k: 'CLASSICAL' for k in
                  ('CLEBSCH', 'CAYLEY', 'KUMMER', 'BARTH', 'TOGLIATTI',
                   'HEART', 'DINGDONG', 'CHMUTOV', 'TANGLE', 'MONKEY')}
SURFACE_FAMILY.update({k: 'HAUSER' for (k, _l, _e, _s, _c, _f) in _HAUSER})


def build_algebraic(kind, res, mu=1.3, clip=0.0, scale=1.0, fold=3):
    """Mesh the zero level set of a preset. Returns (verts, tris).
    marching_tets simply leaves the level set open where it crosses
    the sample box, which for the BOX presets is exactly the wanted
    clip. BALL presets sample the bounding cube of the ball and then
    cull triangles whose centroid falls outside it -- an open, even
    rim (masking outside samples to a large positive value instead
    stitches jagged stair-step caps onto the boundary). `fold` is the
    monkey-saddle fold count n (MONKEY preset only)."""
    label, fn, shape, clip_default = PRESETS[kind]
    r = clip if clip > 0.0 else clip_default
    if kind == 'MONKEY':
        n = max(2, int(round(fold)))
        field = lambda X, Y, Z: fn(X, Y, Z, mu, n)
    else:
        field = lambda X, Y, Z: fn(X, Y, Z, mu)
    mst = _toolkit()
    verts, tris = mst.marching_tets(
        field, (-r, -r, -r), (r, r, r), (res, res, res))
    if shape == 'BALL' and len(tris):
        cen = verts[tris].mean(axis=1)
        keep = np.einsum('ij,ij->i', cen, cen) <= r * r
        tris = tris[keep]
        used = np.unique(tris)               # drop orphaned verts
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        verts = verts[used]
        tris = remap[tris]
    # center on the origin and fit within a 2 m cube, then apply scale
    if len(verts):
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        ext = float((hi - lo).max())
        verts = (verts - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9
                                             else 1.0)
    return verts * scale, tris


def _selftest():
    """Three gates, in increasing cost.

    1. Every Hauser lambda agrees with the caption printed beside it.
       Both are written from the same gallery cell, so this does not
       validate the source -- it stops the two copies drifting apart,
       which is the failure that would otherwise ship a wrong surface
       silently (a mistyped exponent raises nothing; it just moves the
       zero set).
    2. Every preset is reachable from exactly one family.
    3. Every preset meshes to a solid, finite, well-proportioned patch
       inside the 2 m cube.  Aspect ratio is checked because a bad clip
       collapses a surface to a slab while leaving the face count
       healthy -- topology alone would not notice.
    """
    ok = True
    rng = np.random.default_rng(20260820)
    pts = rng.uniform(-1.1, 1.1, size=(3, 400))

    bad = []
    for key, label, eq, _shape, _clip, fn in _HAUSER:
        expr = _printed_expr(eq)
        try:
            ref = eval('lambda x, y, z: ' + expr,
                       {"__builtins__": {}}, {})
            a = np.asarray(fn(pts[0], pts[1], pts[2], 1.3), dtype=float)
            b = np.asarray(ref(pts[0], pts[1], pts[2]), dtype=float)
            scale = max(1.0, float(np.max(np.abs(b))))
            dev = float(np.max(np.abs(a - b))) / scale
            if not np.isfinite(dev) or dev > 1e-12:
                bad.append('%s:%.1e' % (key, dev))
        except Exception as exc:                     # pragma: no cover
            bad.append('%s:%s' % (key, exc))
    ok &= not bad
    print("algebraic: %d Hauser lambdas match their printed equation %s"
          % (len(_HAUSER), 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    orphan = sorted(set(PRESETS) - set(SURFACE_FAMILY))
    fams = {f for f, _ in FAMILIES}
    stray = sorted(f for f in set(SURFACE_FAMILY.values()) if f not in fams)
    good = not orphan and not stray
    ok &= good
    print("algebraic: %d presets all mapped to a declared family %s"
          % (len(PRESETS), 'OK' if good else
             'FAIL orphan=%s stray=%s' % (orphan, stray)))

    thin, empty = [], []
    for key in PRESETS:
        V, T = build_algebraic(key, 40)
        if len(V) < 100 or len(T) < 200 or not np.all(np.isfinite(V)):
            empty.append(key)
            continue
        ext = V.max(axis=0) - V.min(axis=0)
        if float(ext.max()) > 2.0 + 1e-6:
            empty.append(key + '(oversize)')
        elif float(ext.min() / ext.max()) < 0.05:
            thin.append('%s:%.3f' % (key, ext.min() / ext.max()))
    good = not thin and not empty
    ok &= good
    print("algebraic: %d presets mesh solid, finite, in the 2 m cube %s"
          % (len(PRESETS), 'OK' if good else
             'FAIL empty=%s thin=%s' % (empty, thin)))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("algebraic surfaces self-test failed")

