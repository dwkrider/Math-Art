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

try:
    from .. import geom_cache as _geom_cache
except ImportError:  # flat import outside the package
    import geom_cache as _geom_cache


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


# ======================================================================
# Record nodal surfaces
# ======================================================================
# The four surfaces carrying the largest known number of ordinary double
# points for their degree.  Together with the Barth sextic already in
# the classical set above (65 nodes, the maximum for a sextic), these
# are the headline objects of the subject:
#
#   degree  7   Labs septic      99 nodes   (max known; bound is 104)
#   degree  8   Endrass octic   168 nodes   (max known; bound is 174)
#   degree 10   Barth decic     345 nodes   (45 of them at infinity)
#   degree 12   Sarti dodecic   600 nodes   -- NOT SHIPPED, see below
#
# A wrong coefficient in any of these does NOT raise -- it produces a
# surface that renders perfectly well and simply has the wrong number of
# nodes, destroying the one property that makes the object interesting.
# So `_selftest` counts the nodes numerically rather than trusting the
# transcription, calibrated on the shipped Barth sextic (65).
#
# References:
# - O. Labs, "A Septic with 99 real Nodes", arXiv:math/0409348 (2004);
#   Rend. Sem. Mat. Univ. Padova 116 (2006) 299-313.
# - S. Endrass, "A Projective Surface of Degree Eight with 168 Nodes",
#   arXiv:alg-geom/9507011 (1995); J. Algebraic Geom. 6 (1997) 325-334.
# - W. Barth, "Two projective surfaces with many nodes admitting the
#   symmetries of the icosahedron", J. Algebraic Geom. 5 (1996) 173-186
#   -- the decic.  Barth's paper is not freely available; the equation
#   used here is the one published in Wolfram MathWorld ("Barth Decic")
#   and in the AMS Visual Insight entry of 2016-07-01, which agree.
# - A. Sarti, "Pencils of symmetric surfaces in P_3",
#   arXiv:math/0106080; J. Algebra 246 (2001) 429-452 -- the dodecic is
#   the member of the degree-12 pencil at lambda = -22/243.
# Converted copies of the freely available papers, with the
# constructions written out, are in research/papers/algebraic-surfaces/.


# The septic's parameters are algebraic, not rational: they are
# polynomials in the real root of 7a^3 + 7a + 1 (Labs eq. 10).  Solving
# for it beats hard-coding the paper's printed 8 digits.
def _labs_alpha():
    r = np.roots([7.0, 0.0, 7.0, 1.0])
    real = r[np.abs(r.imag) < 1e-12].real
    return float(real[np.argmin(np.abs(real))])


_LABS_ALPHA = _labs_alpha()


def _labs_params():
    """(a1..a5) of Labs section 4, evaluated at the real alpha."""
    a = _LABS_ALPHA
    a2 = a * a
    return (a ** 7 + 7 * a ** 5 - a ** 4 + 7 * a ** 3 - 2 * a2 - 7 * a - 1,
            (a2 + 1) * (3 * a ** 5 + 14 * a ** 3 - 3 * a2 + 7 * a - 3),
            (a2 + 1) ** 2 * (3 * a ** 3 + 7 * a - 3),
            (a * (1 + a2) - 1) * (1 + a2) ** 2,
            -a2 / (1 + a2))


_LABS_A = _labs_params()


def _f_labs(x, y, z, mu):
    # S = P - U in the affine chart w = 1, with a6 = a7 = 1.
    a1, a2, a3, a4, a5 = _LABS_A
    r2 = x * x + y * y
    P = (x * (x ** 6 - 21.0 * x ** 4 * y ** 2 + 35.0 * x ** 2 * y ** 4
              - 7.0 * y ** 6)
         + 7.0 * z * (r2 ** 3 - 8.0 * z ** 2 * r2 ** 2
                      + 16.0 * z ** 4 * r2)
         - 64.0 * z ** 7)
    U = (z + a5) * (a1 * z ** 3 + a2 * z ** 2 + a3 * z + a4
                    + (z + 1.0) * r2) ** 2
    return P - U


def _f_endrass(x, y, z, mu):
    # F = P - Q in the chart w = 1, with c = f = h = 0 and e = -1 (the
    # first two of Endrass's three steps) and the step-3 parameters.
    s2 = _SQRT2
    a = -0.25 * (1.0 + s2)
    b = 0.5 * (2.0 + s2)
    d = (2.0 + 7.0 * s2) / 8.0
    g = 0.5 * (1.0 - 2.0 * s2)
    i = -(1.0 + 12.0 * s2) / 16.0
    r2 = x * x + y * y
    P = (0.25 * (x * x - 1.0) * (y * y - 1.0)
         * ((x + y) ** 2 - 2.0) * ((x - y) ** 2 - 2.0))
    Q = (a * r2 * r2 + r2 * (b * z * z + d)
         - z ** 4 + g * z * z + i) ** 2
    return P - Q


def _f_barth_decic(x, y, z, mu):
    # Chart w = 1.  Phi is the golden ratio; 45 of the 345 nodes lie at
    # infinity, so only 300 are visible here.
    p = _PHI
    p4 = p ** 4
    x2, y2, z2 = x * x, y * y, z * z
    q = x2 + y2 + z2
    return (8.0 * (x2 - p4 * y2) * (y2 - p4 * z2) * (z2 - p4 * x2)
            * (x2 * x2 + y2 * y2 + z2 * z2
               - 2.0 * x2 * y2 - 2.0 * x2 * z2 - 2.0 * y2 * z2)
            + (3.0 + 5.0 * p) * (q - 1.0) ** 2
            * (q - (2.0 - p) ** 2))


_SARTI_PAIRS = [(i, j) for i in range(4) for j in range(4) if i != j]
_SARTI_TRIPLES = [(i, j, k) for i in range(4) for j in range(4)
                  for k in range(4) if len({i, j, k}) == 3]
_SARTI_QUADS = [(i, j, k, h) for i in range(4) for j in range(4)
                for k in range(4) for h in range(4)
                if len({i, j, k, h}) == 4]


def _f_sarti(x, y, z, mu):
    # The lambda = -22/243 member of the degree-12 pencil
    # S12(x) + lambda Q12(x), in the chart x3 = 1.
    #
    # Sarti writes the S_pq... sums "over all the indices i, j, k" with
    # distinct indices wherever they appear together, i.e. over ordered
    # tuples -- which is what is done here.
    ys = (x * x, y * y, z * z, 1.0 + 0.0 * x)
    P2, T3, Q4 = _SARTI_PAIRS, _SARTI_TRIPLES, _SARTI_QUADS
    S51 = sum(ys[i] ** 5 * ys[j] for i, j in P2)
    S42 = sum(ys[i] ** 4 * ys[j] ** 2 for i, j in P2)
    S33 = sum(ys[i] ** 3 * ys[j] ** 3 for i, j in P2)
    S411 = sum(ys[i] ** 4 * ys[j] * ys[k] for i, j, k in T3)
    S321 = sum(ys[i] ** 3 * ys[j] ** 2 * ys[k] for i, j, k in T3)
    S222 = sum(ys[i] ** 2 * ys[j] ** 2 * ys[k] ** 2 for i, j, k in T3)
    S3111 = sum(ys[i] ** 3 * ys[j] * ys[k] * ys[h] for i, j, k, h in Q4)
    S2211 = sum(ys[i] ** 2 * ys[j] ** 2 * ys[h] * ys[k]
                for i, j, k, h in Q4)
    fs = (2.0 * S51 - 6.0 * S42 - 12.0 * S411 + 14.0 * S33
          + 9.0 * S321 + 348.0 * S3111 + 30.0 * S222 - 270.0 * S2211)
    y0, y1, y2, y3 = ys
    fa = (y0 ** 3 * (y1 ** 2 * y2 - y1 * y2 ** 2 + y2 ** 2 * y3
                     - y2 * y3 ** 2 + y3 ** 2 * y1 - y3 * y1 ** 2)
          - y1 ** 3 * (y2 ** 2 * y3 - y2 * y3 ** 2 + y3 ** 2 * y0
                       - y3 * y0 ** 2 + y0 ** 2 * y2 - y0 * y2 ** 2)
          + y2 ** 3 * (y0 ** 2 * y1 - y0 * y1 ** 2 + y1 ** 2 * y3
                       - y1 * y3 ** 2 + y3 ** 2 * y0 - y3 * y0 ** 2)
          - y3 ** 3 * (y0 ** 2 * y1 - y0 * y1 ** 2 + y1 ** 2 * y2
                       - y1 * y2 ** 2 + y2 ** 2 * y0 - y2 * y0 ** 2))
    S12 = fs + 33.0 * math.sqrt(5.0) * fa
    Q12 = (ys[0] + ys[1] + ys[2] + ys[3]) ** 6
    return S12 - (22.0 / 243.0) * Q12


_RECORD = (
    ('LABS', "Labs Septic (99 nodes)", 'BALL', 2.6, _f_labs),
    ('ENDRASS', "Endrass Octic (168 nodes)", 'BALL', 2.6, _f_endrass),
    ('BARTH_DECIC', "Barth Decic (345 nodes)", 'BALL', 2.6,
     _f_barth_decic),
    # Sarti's degree-12 surface with 600 nodes is NOT here.  `_f_sarti`
    # below transcribes her S12 and the lambda = -22/243 pencil member,
    # and it has the right symmetry -- but it could not be verified, so
    # shipping it under that name would be a claim we cannot support.
    # What fails: Sarti fixes the singular members of the pencil by the
    # orbits they contain, so at an orbit point p the ratio
    # -S12(p)/Q12(p) must be one of her four values (-3/32, -22/243,
    # -2/25, 0).  Evaluated at 600-cell vertices it lands on none of
    # them, under either reading of her "sums over distinct indices"
    # (ordered or unordered tuples).  Either the summation
    # multiplicities differ from both readings, or the orbit points need
    # her frame -- she places the polytopes as in Coxeter p.157 *with x0
    # and x1 interchanged*, which is not reproduced here.  See BACKLOG.
)

for _key, _label, _shape, _clip, _fn in _RECORD:
    PRESETS[_key] = (_label, _fn, _shape, _clip)


# ======================================================================
# Named implicit surfaces from the 3D-XplorMath / Virtual Math Museum
# ======================================================================
# Compact level sets chosen for their TOPOLOGY: each is a thickened tube
# around a curve or a union of curves, so the genus is the point of the
# surface rather than a by-product.  Unlike everything above they are
# levels f = ff, not zero sets, so `ff` is a real parameter -- pushing
# it up eventually fattens the tube until the holes close and the genus
# collapses.
#
# `_selftest` measures the genus of every row from the meshed Euler
# characteristic and compares it against the value the source documents.
# That is a much sharper gate than "it meshed": it is the defining
# property of these surfaces, it is sensitive to both the formula and
# the level, and it is what would break first if either were wrong.
#
# References:
# - 3DXM Consortium, "Functions with compact levels in 3D-XplorMath",
#   virtualmathmuseum.org/docs/ImplicitCompact.pdf -- the formulas and
#   the default parameters used here.  A converted copy is in
#   research/papers/algebraic-surfaces/vmm-implicit-surfaces-compact/.
# - H. Karcher and R. Palais, 3D-XplorMath / Virtual Math Museum
#   surface gallery, virtualmathmuseum.org/Surface/.
#
# One correction to the source, deliberate: the reference calls
# Orthocircles "a genus 5 tube around three intersecting orthogonal
# circles".  It is genus 7.  Three unit circles, one per coordinate
# plane, meet pairwise in 2 points, so the graph they form has V = 6 and
# E = 12 (each circle carries 4 of the points and is cut into 4 arcs),
# giving first Betti number E - V + 1 = 7; a tube around it has that
# genus.  The mesh measures 7, so the gate below expects 7 and the
# source's 5 is taken to be a slip.


def _f_vmm_pretzel(x, y, z, mu):
    # f := h^2 + (1+cc) z^2,  h the quotient of the two circle factors
    cc = 1.0
    num = ((x - cc) ** 2 + y * y - 1.0) * ((x + cc) ** 2 + y * y - 1.0)
    h = num / (1.0 + (1.0 + cc) * (x * x + y * y))
    return h * h + (1.0 + cc) * z * z - 0.05


def _f_vmm_bretzel2(x, y, z, mu):
    # genus 2 tube around a figure eight; genus 0 for large ff
    bb = 1.0
    num = ((1.0 - x * x) * x * x - y * y) ** 2 + z * z / 2.0
    return num / (1.0 + bb * (x * x + y * y + z * z)) - 0.025


def _f_vmm_bretzel5(x, y, z, mu):
    # genus 5 tube around two intersecting ellipses
    return (((x * x + y * y / 4.0 - 1.0)
             * (x * x / 4.0 + y * y - 1.0)) ** 2
            + z * z / 2.0 - 0.075)


def _f_vmm_pilz(x, y, z, mu):
    # genus 3 tube around a circle and an orthogonal ellipse
    aa, bb, cc, dd = 1.0, 1.0, 0.03, 0.28
    return (((x * x + y * y - 1.0) ** 2 + (z - 0.5) ** 2)
            * ((y * y / (aa * aa) + (z + cc) ** 2 - 1.0) ** 2 + x * x)
            - dd * dd * (1.0 + bb * (z - 0.5) ** 2))


def _f_vmm_orthocircles(x, y, z, mu):
    # tube around the three unit circles of the coordinate planes
    aa, ff = 1.0, 0.05
    return (((x * x / aa + y * y - 1.0) ** 2 + z * z)
            * ((y * y / aa + z * z - 1.0) ** 2 + x * x)
            * ((z * z / aa + x * x - 1.0) ** 2 + y * y) - ff)


def _f_vmm_decocube(x, y, z, mu):
    # tube around six circles of radius cc on the faces of a cube
    cc, ff = 0.8, 0.02
    c2 = cc * cc
    return (((x * x + y * y - c2) ** 2 + (z * z - 1.0) ** 2)
            * ((y * y + z * z - c2) ** 2 + (x * x - 1.0) ** 2)
            * ((z * z + x * x - c2) ** 2 + (y * y - 1.0) ** 2) - ff)


def _f_vmm_join2tori(x, y, z, mu):
    # two tori joined by a handle; below ff ~ 0.1 they come apart again
    aa, bb, cc, ff = 1.0, 0.35, 1.1, 0.2
    r2 = y * y + z * z
    tor_r = ((x - cc) ** 2 + r2 - aa * aa - bb * bb) ** 2 \
        + 4.0 * aa * aa * (z * z - bb * bb)
    tor_l = ((x + cc) ** 2 + r2 - aa * aa - bb * bb) ** 2 \
        + 4.0 * aa * aa * (z * z - bb * bb)
    den = 1.0 + (x - cc) ** 2 + (x + cc) ** 2 + y * y + z * z / 2.0
    return tor_r * tor_l / den - ff


# (key, label, documented genus, clip shape, clip, F)
_NAMED = (
    ('VMM_PRETZEL', "Pretzel", 2, 'BOX', 3.0, _f_vmm_pretzel),
    ('VMM_BRETZEL2', "Bretzel2", 2, 'BOX', 1.6, _f_vmm_bretzel2),
    ('VMM_BRETZEL5', "Bretzel5", 5, 'BOX', 2.6, _f_vmm_bretzel5),
    ('VMM_PILZ', "Pilz Surface", 3, 'BOX', 2.0, _f_vmm_pilz),
    ('VMM_ORTHOCIRCLES', "Orthocircles", 7, 'BOX', 1.6,
     _f_vmm_orthocircles),
    ('VMM_DECOCUBE', "Deco-Cube", 13, 'BOX', 1.6, _f_vmm_decocube),
    ('VMM_JOIN2TORI', "Join of Two Tori", 2, 'BOX', 3.0,
     _f_vmm_join2tori),
)

NAMED_GENUS = {k: g for (k, _l, g, _s, _c, _f) in _NAMED}

for _key, _label, _g, _shape, _clip, _fn in _NAMED:
    PRESETS[_key] = (_label, _fn, _shape, _clip)


# ======================================================================
# Named surfaces from Ferreol's Encyclopedie des formes mathematiques
# ======================================================================
# Four classical implicit surfaces, each carrying a property that is
# exactly checkable and that IS the reason the surface has a name.  The
# self-test gates on that property rather than on "it meshed", because a
# mistyped exponent moves the zero set without raising anything:
#
#   Cartan's umbrella   an ISOLATED LINE.  Every point of Oz satisfies
#       z(x^2+y^2) = x^3, yet no neighbourhood of a point of Oz other
#       than the origin meets the rest of the surface -- the handle of
#       the umbrella.  Gated by evaluating on Oz and by confirming the
#       canopy is the cone over an Agnesi cubic.
#   Cassini surface     its LEVEL CURVES ARE CASSINI OVALS.  The level
#       z is the oval with parameter b = z, so the section y = 0 splits
#       into the circle x^2+z^2 = a^2 and the conjugate hyperbola
#       x^2-z^2 = a^2.  Both are gated exactly.
#   Cassini of revolution   the BIFOCAL equation r r' = b^2 with foci at
#       (+-a, 0, 0).  Gated by measuring the two distances.
#   Titeica's surface   the affine sphere xyz = a^3: the Gauss curvature
#       at M is proportional to the fourth power of the distance from a
#       fixed centre to the tangent plane at M.  Gated by computing K
#       and d from the graph z = a^3/(xy) and confirming d^4/K = 27a^6
#       is genuinely constant.  (The source prints 26a^6, which does not
#       follow from the K and d it prints on the same line; 27 does, and
#       27 is what the surface has.)
#
# References:
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#   mathcurve.com -- the chapters "parapluie de Cartan", "surface de
#   Cassini", "surface de Titeica".  A converted copy of the whole
#   encyclopedia is in research/books/
#   mathcurve_encyclopedie_formes_mathematiques/.
# - H. Cartan and H. Whitney, 1957 -- the umbrella; cf. Whitney's own
#   umbrella x^2 = y^2 z, already in the Hauser block above.
# - G. D. Cassini, the ovals of 1680; the surface is the classical
#   three-dimensional generalisation of them.
# - G. Titeica, "Sur une nouvelle classe de surfaces", Rend. Circ. Mat.
#   Palermo 25 (1908) 180-187 -- the affine spheres of 1907.


def _f_cartan(x, y, z, mu):
    # z(x^2+y^2) = x^3.  The cone over an Agnesi cubic, with the whole
    # of Oz in the zero set but isolated from the canopy.
    return z * (x * x + y * y) - x * x * x


def _f_cassini(x, y, z, mu):
    # ((x+a)^2+y^2)((x-a)^2+y^2) = z^4: the surface whose horizontal
    # sections are the Cassini ovals of parameter b = z.
    a = 1.0
    return (((x + a) ** 2 + y * y) * ((x - a) ** 2 + y * y) - z ** 4)


def _f_cassini_rev(x, y, z, mu):
    # (x^2+y^2+z^2+a^2)^2 = b^4 + 4a^2 x^2: a Cassini oval revolved
    # about its long axis.  b is taken in the window a < b < a sqrt2,
    # where the surface is a single peanut rather than two lobes.
    a, b = 1.0, 1.2
    return (x * x + y * y + z * z + a * a) ** 2 - b ** 4 - 4.0 * a * a * x * x


def _f_titeica(x, y, z, mu):
    # xyz = a^3, the cubic affine sphere.
    a = 1.0
    return x * y * z - a ** 3


_MATHCURVE = (
    ('CARTAN', "Cartan's Umbrella", 'BOX', 1.4, _f_cartan),
    ('CASSINI', "Cassini Surface", 'BOX', 1.7, _f_cassini),
    ('CASSINI_REV', "Cassini Surface of Revolution", 'BOX', 1.2,
     _f_cassini_rev),
    ('TITEICA', "Titeica Surface", 'BOX', 2.6, _f_titeica),
)

for _key, _label, _shape, _clip, _fn in _MATHCURVE:
    PRESETS[_key] = (_label, _fn, _shape, _clip)


# ======================================================================
# Goursat's surfaces: a Platonic symmetry group, at each degree
# ======================================================================
# These arrived as an operator of their own and were folded in here,
# because they are the same kind of object as everything above -- the
# zero set of one polynomial, meshed by the same marching tetrahedra,
# clipped and rimmed by the same controls -- and because keeping them
# apart HID three duplicates.  Goursat's tetrahedral cubic at k = 4,
# rescaled by 4a, is exactly the Cayley preset above; his k' = 0 member
# is exactly the Titeica one; and his dodecahedral (0, 5/4, -5/2, 5/4)
# is Barth's sextic in a different frame.  In one catalogue those read
# as the same surface reached two ways, which is true and is the
# interesting part; across two operators they just looked like padding.
#
# They differ from every other row here in ONE way: they are FAMILIES,
# continuous in their coefficients, so the interest is partly in
# dragging them.  That is what `PRESET_PARAMS` below is for.
#
# The mathematics, the coefficient tables and the gates stay in
# `math_art/surfaces/goursat.py`.


def _goursat():
    try:
        from . import goursat as g
    except ImportError:                  # flat import outside the package
        import goursat as g
    return g


def _goursat_field(family, defaults):
    """A field for one Goursat row, with its coefficients LIVE.

    The row's own tuple is the default; the operator may override any of
    them.  Written as a closure per row rather than a branch in
    `build_algebraic` so that a Goursat preset is an ordinary row of
    `PRESETS` and nothing downstream has to know it is special.
    """
    n = len(defaults)

    def field(x, y, z, mu, k0=None, k1=None, k2=None, k3=None, size=1.0):
        ks = (k0, k1, k2, k3)
        co = tuple(defaults[i] if ks[i] is None else float(ks[i])
                   for i in range(n))
        return _goursat().FAMILY_FIELD[family](x, y, z, co, float(size))

    return field


#: Goursat family key -> (algebraic family key, label)
_GOURSAT_FAMILY = {
    'OCT4': ('GOURSAT_OCT', "Goursat (Octahedral)"),
    'TET3': ('GOURSAT_TET', "Goursat (Tetrahedral)"),
    'TET4': ('GOURSAT_TET', "Goursat (Tetrahedral)"),
    'DODEC6': ('GOURSAT_DODEC', "Goursat (Dodecahedral)"),
}

#: what each Goursat family calls its coefficients, in order
GOURSAT_COEFF_NAMES = {
    'OCT4': ("k", "k'", "k''"),
    'TET3': ("k", "k'"),
    'TET4': (),
    'DODEC6': ("k", "k'", "k''", "k'''"),
}

GOURSAT_PRESETS = {}


def _load_goursat():
    """Fold every Goursat row into PRESETS.  Done in a function so the
    import stays lazy and this module still loads if the engine is
    absent (the flat-import fallbacks elsewhere rely on that)."""
    g = _goursat()
    rows = {}
    for key in g.PRESET_ORDER:
        label, fam, co, clip, desc = g.PRESETS[key]
        afam, _flabel = _GOURSAT_FAMILY[fam]
        PRESETS[key] = (label, _goursat_field(fam, co), 'BALL', clip)
        SURFACE_FAMILY[key] = afam
        rows[key] = (fam, co, desc)
    return rows


# Which family each preset belongs to.  The operator turns this into a
# Family dropdown that filters the Surface list: an 80-entry flat enum
# is unusable, and the families are how the sources group them anyway.
FAMILIES = (
    ('CLASSICAL', "Classical"),
    ('HAUSER', "Hauser Gallery"),
    ('RECORD', "Record Nodal Surfaces"),
    ('NAMED', "Named Implicit Surfaces"),
    ('MATHCURVE', "Encyclopedia Surfaces"),
    ('GOURSAT_OCT', "Goursat (Octahedral)"),
    ('GOURSAT_TET', "Goursat (Tetrahedral)"),
    ('GOURSAT_DODEC', "Goursat (Dodecahedral)"),
)

# Sampling resolution each family wants by default.
#
# 120 throughout.  It was 80 in general and 120 only for the record nodal
# surfaces, on the grounds that those are degree 7 to 10 with hundreds of
# double points packed into a small region and turn to mush at 80.  That
# reasoning was sound but too narrow: the Hauser gallery is full of
# surfaces with cusps, self-intersections and thin sheets that 80 also
# blurs, and a surface whose whole interest is a singularity is the rule
# in this table rather than the exception.  Per-family overrides stay
# available for anything that later turns out to want more.
FAMILY_RESOLUTION_DEFAULT = 120
FAMILY_RESOLUTION = {}


SURFACE_FAMILY = {k: 'CLASSICAL' for k in
                  ('CLEBSCH', 'CAYLEY', 'KUMMER', 'BARTH', 'TOGLIATTI',
                   'HEART', 'DINGDONG', 'CHMUTOV', 'TANGLE', 'MONKEY')}
SURFACE_FAMILY.update({k: 'HAUSER' for (k, _l, _e, _s, _c, _f) in _HAUSER})
SURFACE_FAMILY.update({k: 'RECORD' for (k, _l, _s, _c, _f) in _RECORD})
SURFACE_FAMILY.update({k: 'NAMED' for (k, _l, _g, _s, _c, _f) in _NAMED})
SURFACE_FAMILY.update({k: 'MATHCURVE' for (k, _l, _s, _c, _f)
                       in _MATHCURVE})

GOURSAT_PRESETS.update(_load_goursat())


# ======================================================================
# per-preset parameters
# ======================================================================
# Almost every preset here is a fixed zero set and takes nothing.  A few
# are families with live parameters, and each declares what it takes
# rather than the operator carrying a hard-coded branch per such preset.
#
# That is not new machinery for Goursat's sake -- it RETIRES the two
# branches that were already there, the Kummer quartic's node-sharpness
# and the monkey saddle's fold count, which the operator drew and passed
# by name.  The operator now draws exactly what a row declares and
# forwards it, so the next parameterised surface costs a table row.
#
# Row: (attribute, label, kind, default, lo, hi, description)
PRESET_PARAMS = {
    'KUMMER': (
        ('mu', "Node Sharpness", 'FLOAT', 1.3, 1.05, 2.0,
         "Kummer quartic parameter; it moves the sixteen nodes"),),
    'MONKEY': (
        ('fold', "Folds", 'INT', 3, 2, 8,
         "Saddle fold count: 2 is the ordinary saddle, 3 the monkey "
         "saddle, higher the n-fold saddles"),),
}

_GOURSAT_LIMITS = ((-40.0, 40.0), (-40.0, 40.0),
                   (-100.0, 100.0), (-200.0, 200.0))

for _k, (_fam, _co, _desc) in GOURSAT_PRESETS.items():
    _rows = []
    for _i, _nm in enumerate(GOURSAT_COEFF_NAMES[_fam]):
        _lo, _hi = _GOURSAT_LIMITS[_i]
        _rows.append(('k%d' % _i, _nm, 'FLOAT', float(_co[_i]), _lo, _hi,
                      "Coefficient %s of the family; the surface is "
                      "continuous in it, so dragging it walks the "
                      "family through its transitions" % _nm))
    _rows.append(
        ('size', "Size", 'FLOAT', 1.0, 0.05, 10.0,
         "The length a the coefficients are measured against. It "
         "rescales the surface inside the clip ball rather than the "
         "finished object, so it changes how much is shown"))
    PRESET_PARAMS[_k] = tuple(_rows)


def preset_params(kind):
    """The parameter rows a preset declares, or ()."""
    return PRESET_PARAMS.get(kind, ())


def boundary_loops(verts, tris, smooth=2):
    """Ordered polylines along the open edge -- see math_art/rim_curve.

    Kept as a name here because this module's `_selftest` gates on it
    and several callers import it from the engine, but the
    implementation is shared with every other generator that offers a
    rim so the behaviour cannot drift between them.
    """
    try:
        from .. import rim_curve
    except ImportError:
        import rim_curve
    return rim_curve.boundary_loops(verts, tris, smooth=smooth)


@_geom_cache.memoise(version=1)
def build_algebraic(kind, res, mu=1.3, clip=0.0, scale=1.0, fold=3,
                    **params):
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
    # A row that declares a `size` measures its clip radius in units of
    # it, so the window follows the surface instead of cropping it.
    r = r * float(params.get('size', 1.0))
    if kind == 'MONKEY':
        n = max(2, int(round(params.get('fold', fold))))
        field = lambda X, Y, Z: fn(X, Y, Z, mu, n)
    elif params:
        field = lambda X, Y, Z: fn(X, Y, Z, mu, **params)
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

    # The record surfaces are defined by symmetry, and their published
    # node counts depend on every coefficient being exact.  A node count
    # is the ideal gate but is not reliable to compute numerically; the
    # SYMMETRY is, and it is an exact algebraic identity that almost any
    # coefficient typo destroys.  The septic's 7-fold invariance, for
    # instance, only holds if the expansion 21/35/7/8/16/64 of its
    # product of seven planes is right.
    def _rot(fn, n, px, py, pz):
        t = 2.0 * math.pi / n
        return fn(px * math.cos(t) - py * math.sin(t),
                  px * math.sin(t) + py * math.cos(t), pz, 1.3)

    qx, qy, qz = rng.uniform(-1.5, 1.5, size=(3, 300))

    def _dev(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        sc = max(float(np.max(np.abs(a))), 1e-30)
        return float(np.max(np.abs(a - b))) / sc

    checks = [
        ("Labs septic D7", _dev(_f_labs(qx, qy, qz, 1.3),
                                _rot(_f_labs, 7, qx, qy, qz))),
        ("Labs septic mirror", _dev(_f_labs(qx, qy, qz, 1.3),
                                    _f_labs(qx, -qy, qz, 1.3))),
        ("Endrass octic D8", _dev(_f_endrass(qx, qy, qz, 1.3),
                                  _rot(_f_endrass, 8, qx, qy, qz))),
        ("Endrass octic mirror", _dev(_f_endrass(qx, qy, qz, 1.3),
                                      _f_endrass(qx, -qy, qz, 1.3))),
        ("Barth decic 3-cycle", _dev(_f_barth_decic(qx, qy, qz, 1.3),
                                     _f_barth_decic(qy, qz, qx, 1.3))),
        ("Barth sextic 3-cycle", _dev(_f_barth(qx, qy, qz, 1.3),
                                      _f_barth(qy, qz, qx, 1.3))),
    ]
    worst = [f"{n}:{d:.1e}" for n, d in checks if not (d < 1e-12)]
    ok &= not worst
    print("algebraic: %d record-surface symmetry identities hold %s"
          % (len(checks), 'OK' if not worst else 'FAIL ' + ','.join(worst)))

    # The septic's parameters are polynomials in the real root of
    # 7a^3+7a+1; Labs prints it as -0.14010685, so a solver that drifted
    # would show up here.
    good = abs(_LABS_ALPHA - (-0.14010685)) < 5e-9
    ok &= good
    print("algebraic: Labs alpha = %.10f vs published -0.14010685 %s"
          % (_LABS_ALPHA, 'OK' if good else 'FAIL'))

    # These are tubes around curves, so the genus IS the surface.  Check
    # it from the meshed Euler characteristic rather than trusting the
    # level: too large an ff fattens the tube until holes close and the
    # genus silently drops (the source documents exactly that for
    # Bretzel2), which no other gate here would notice.
    def _genus(V, T):
        e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        e = np.sort(e, axis=1)
        u, cnt = np.unique(e, axis=0, return_counts=True)
        if int((cnt != 2).sum()):
            return None                      # not closed; genus undefined
        parent = list(range(len(V)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in u:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
        comps = len({find(i) for i in range(len(V))})
        chi = len(V) - len(u) + len(T)
        return (2 * comps - chi) // 2

    wrong = []
    for key, want in NAMED_GENUS.items():
        Vg, Tg = build_algebraic(key, 96)
        got = _genus(Vg, Tg) if len(Tg) else None
        if got != want:
            wrong.append('%s:%s!=%d' % (key, got, want))
    ok &= not wrong
    print("algebraic: %d named implicit surfaces have their documented "
          "genus %s" % (len(NAMED_GENUS),
                        'OK' if not wrong else 'FAIL ' + ','.join(wrong)))

    # The rim curve sweeps the open edge, so it must find exactly the
    # boundary edges and nothing else.  Checked two ways: a closed
    # surface must yield no loops at all, and on an open one the edges
    # of the returned chains must be precisely the once-used edges.
    def _rim_ok(key):
        V, T = build_algebraic(key, 48)
        e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        e = np.sort(e, axis=1)
        u, c = np.unique(e, axis=0, return_counts=True)
        nb = int((c == 1).sum())
        loops = boundary_loops(V, T, smooth=0)
        got = sum(len(p) if closed else len(p) - 1 for p, closed in loops)
        return nb, got

    bad = []
    for key in ('CLEBSCH', 'KREUZ', 'LABS'):
        nb, got = _rim_ok(key)
        if nb == 0 or got != nb:
            bad.append('%s:%d/%d' % (key, got, nb))
    for key in ('VMM_ORTHOCIRCLES', 'VMM_BRETZEL5'):
        V, T = build_algebraic(key, 48)
        if boundary_loops(V, T):
            bad.append(key + ':closed-surface-has-rim')
    ok &= not bad
    print("algebraic: rim curve spans exactly the open edge %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # The encyclopedia surfaces are named for a property, not for a
    # shape, so each is gated on its own property.  Every one of these
    # is an exact identity that a mistyped exponent destroys while
    # leaving a perfectly meshable surface behind.
    checks = []

    # Cartan: the whole of Oz lies on it, and the canopy is the cone
    # over the Agnesi cubic -- i.e. the cartesian and the parametric
    # forms agree, (u cos v, u sin v, u cos^3 v).
    zs = rng.uniform(-3.0, 3.0, 200)
    checks.append(("Cartan umbrella contains the isolated line Oz",
                   float(np.max(np.abs(_f_cartan(0.0 * zs, 0.0 * zs, zs,
                                                 1.3))))))
    uu = rng.uniform(0.2, 2.0, 200)
    vv = rng.uniform(0.0, 2.0 * math.pi, 200)
    checks.append(("Cartan umbrella is the cone over an Agnesi cubic",
                   float(np.max(np.abs(_f_cartan(
                       uu * np.cos(vv), uu * np.sin(vv),
                       uu * np.cos(vv) ** 3, 1.3))))))

    # Cassini: the section y = 0 is the circle x^2+z^2 = a^2 united
    # with the conjugate hyperbola x^2-z^2 = a^2 -- both exactly.
    tt = rng.uniform(0.0, 2.0 * math.pi, 200)
    checks.append(("Cassini section y=0 contains the circle x^2+z^2=a^2",
                   float(np.max(np.abs(_f_cassini(np.cos(tt), 0.0 * tt,
                                                  np.sin(tt), 1.3))))))
    tt = rng.uniform(-1.2, 1.2, 200)
    checks.append(("Cassini section y=0 contains x^2-z^2=a^2",
                   float(np.max(np.abs(_f_cassini(np.cosh(tt), 0.0 * tt,
                                                  np.sinh(tt), 1.3))))))
    # and the level curve at height z IS the Cassini oval of parameter
    # b = z: the product of the distances to (+-a, 0) equals z^2.
    px, py = rng.uniform(-2.0, 2.0, (2, 200))
    d1 = np.hypot(px + 1.0, py)
    d2 = np.hypot(px - 1.0, py)
    lvl = np.sqrt(d1 * d2)                   # the height whose oval this is
    checks.append(("Cassini level curves are Cassini ovals rr' = z^2",
                   float(np.max(np.abs(_f_cassini(px, py, lvl, 1.3))))
                   / float(np.max(d1 * d2)) ** 2))

    # Cassini of revolution: the bifocal equation r r' = b^2.
    a_, b_ = 1.0, 1.2
    th = rng.uniform(0.0, math.pi, 300)
    ph = rng.uniform(0.0, 2.0 * math.pi, 300)
    # The polar form about the focal axis Ox is r^4 - 2a^2 r^2 cos2t
    # + a^4 = b^4, so solve it for r on each ray.
    ct = np.cos(th)
    c2t = 2.0 * ct * ct - 1.0                # cos 2t
    disc = b_ ** 4 - a_ ** 4 * (1.0 - c2t * c2t)
    r2 = a_ * a_ * c2t + np.sqrt(np.maximum(disc, 0.0))
    keep = r2 > 1e-9
    r = np.sqrt(r2[keep])
    px = r * ct[keep]
    py = r * np.sin(th[keep]) * np.cos(ph[keep])
    pz = r * np.sin(th[keep]) * np.sin(ph[keep])
    rr = (np.sqrt((px + a_) ** 2 + py * py + pz * pz)
          * np.sqrt((px - a_) ** 2 + py * py + pz * pz))
    checks.append(("Cassini of revolution obeys the bifocal rr' = b^2",
                   float(np.max(np.abs(rr - b_ * b_)))))
    checks.append(("Cassini of revolution: those points are on it",
                   float(np.max(np.abs(_f_cassini_rev(px, py, pz, 1.3))))))

    # Titeica: on the graph z = a^3/(xy) the Gauss curvature K and the
    # distance d from the origin to the tangent plane satisfy
    # d^4 / K = 27 a^6 -- constant, which is the DEFINITION of a
    # Titeica surface (the source prints 26; 27 is what its own K and d
    # give, and what the surface has).
    a_ = 1.0
    px, py = rng.uniform(0.6, 2.2, (2, 300))
    pz = a_ ** 3 / (px * py)
    p_ = -a_ ** 3 / (px ** 2 * py)           # z_x
    q_ = -a_ ** 3 / (px * py ** 2)           # z_y
    r_ = 2.0 * a_ ** 3 / (px ** 3 * py)      # z_xx
    s_ = a_ ** 3 / (px ** 2 * py ** 2)       # z_xy
    t_ = 2.0 * a_ ** 3 / (px * py ** 3)      # z_yy
    Kt = (r_ * t_ - s_ * s_) / (1.0 + p_ * p_ + q_ * q_) ** 2
    dt = np.abs(p_ * px + q_ * py - pz) / np.sqrt(p_ * p_ + q_ * q_ + 1.0)
    ratio = dt ** 4 / Kt
    checks.append(("Titeica surface has d^4/K = 27 a^6 constant",
                   float(np.max(np.abs(ratio - 27.0 * a_ ** 6)))
                   / (27.0 * a_ ** 6)))
    checks.append(("Titeica: those graph points satisfy xyz = a^3",
                   float(np.max(np.abs(_f_titeica(px, py, pz, 1.3))))))

    worst = ['%s:%.1e' % (n, d) for n, d in checks if not (d < 1e-10)]
    ok &= not worst
    print("algebraic: %d encyclopedia-surface properties hold %s"
          % (len(checks), 'OK' if not worst else 'FAIL ' + ','.join(worst)))

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

