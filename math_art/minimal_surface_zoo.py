
# Minimal-surface zoo: the data catalog for the Weierstrass-Enneper /
# Bjorling engine in we_builders.py.
#
# Each surface is one table row (Gauss map g, height differential dh
# or an explicit phi / antiderivative stack, a domain, and UI hooks);
# register() wires every row into the toolkit's PARAMETRIC /
# MESH_PARAM / COUNT_PARAM / ANGLE_PARAM registries, so rows surface
# automatically in the mesh.parametric_minimal_add operator.
#
# References:
#   The catalog follows Matthias Weber's repository of minimal
#   surfaces, https://minimalsurfaces.blog/ , and his lecture notes
#   "Classical Minimal Surfaces in Euclidean Space by Examples"
#   (Indiana University) -- gratefully credited.  Specific sources:
#   L. P. Jorge & W. H. Meeks III (1983) k-noids; C. J. Costa (1982),
#   D. Hoffman & W. H. Meeks III (1985+) Costa-Hoffman-Meeks;
#   H. Karcher (1988) saddle towers ("Embedded minimal surfaces
#   derived from Scherk's examples", Manuscripta Math. 62);
#   W. H. Meeks III (1981) the minimal Mobius strip ("The
#   classification of complete minimal surfaces in R3 with total
#   curvature greater than -8pi", Duke Math. J. 48);
#   B. Riemann (1867) the singly periodic minimal example foliated
#   by circles; H. A. Schwarz (1890) the Bjorling formula;
#   E. Catalan (1855), H. F. Scherk (1835), H. W. Richmond (1904).

import math
import numpy as np

try:
    from . import we_builders as we
except ImportError:
    import we_builders as we

TAU = 2.0 * math.pi


# ==========================================================================
# Families (Weber's taxonomy, artist-friendly).  Order = menu order.
# ==========================================================================

FAMILIES = (
    ('CLASSICAL', "Classical"),
    ('SPHERES', "Spheres (genus 0)"),
    ('TORI', "Tori (genus 1)"),
    ('HIGHER', "Higher Genus"),
    ('SINGLY', "Singly Periodic"),
    ('DOUBLY', "Doubly Periodic"),
    ('BJORLING', "Bjorling (curve-seeded)"),
    ('NONORIENT', "Non-Orientable"),
)

# families of the surfaces defined directly in the toolkit
LEGACY_FAMILY = {
    'ENNEPER': 'CLASSICAL', 'CATENOID': 'CLASSICAL',
    'HELICOID': 'CLASSICAL', 'HENNEBERG': 'CLASSICAL',
    'CATALAN': 'CLASSICAL', 'BOUR': 'CLASSICAL',
    'RICHMOND': 'CLASSICAL', 'SCHERK1': 'DOUBLY',
    'CATHEL': 'CLASSICAL',
    'COSTA': 'TORI', 'CHEN_GACK': 'TORI',
}


# ==========================================================================
# Costa-Hoffman-Meeks pieces (ported from the toolkit onto the engine)
# ==========================================================================

def _chm_w(z, k):
    """Single-valued branch of (z^k (z^2-1))^(1/(k+1)) on the upper
    half plane (cuts along [0,1] and (-inf,-1])."""
    return np.exp((k * np.log(z) + np.log(z - 1.0) + np.log(z + 1.0))
                  / (k + 1))


_CHM_C_CACHE = {}


def chm_modulus(k, n=4096):
    """Real modulus c fixing the CHM end balance, from the period
    ratio on an ellipse around the cut [0,1]: c^2 = -Im(A)/Im(B)."""
    if k not in _CHM_C_CACHE:
        A = we.period_integral(
            lambda z: _chm_w(z, k) / (z * z - 1.0), 0.5, 0.75, 0.35, n)
        B = we.period_integral(
            lambda z: 1.0 / (_chm_w(z, k) * (z * z - 1.0)),
            0.5, 0.75, 0.35, n)
        _CHM_C_CACHE[k] = math.sqrt(-A.imag / B.imag)
    return _CHM_C_CACHE[k]


def _chm_phi(z, p):
    k, c = p['k'], p['c']
    w = _chm_w(z, k)
    d = z * z - 1.0
    return (0.5 * (w - c * c / w) / d,
            0.5j * (w + c * c / w) / d,
            c / d)


def _chm_seed(p, z0):
    """Analytic integral of phi over the first spine cell 0 -> z0
    (the integrand ~ z^-(k/(k+1)) at the branch point)."""
    k, c = p['k'], p['c']
    I1 = (c * c / 2) * np.exp(-1j * math.pi / (k + 1)) * (k + 1) \
        * z0 ** (1.0 / (k + 1))
    return (-I1, -1j * I1, -c * z0)


# ==========================================================================
# Riemann's minimal example (torus closed form)
# ==========================================================================
# On the rectangular torus C/<1, tau>, tau = i t:  g = c (wp - e2) with
# e2 = wp((1+tau)/2), dh = dz.  With c = 1 / sqrt((e2-e1)(e2-e3)) all
# periods close except one tilted translation (0, y, 1) -- Riemann's
# one-parameter family of planes joined by necks, foliated by circles.
# Using 1/(wp - e2) = (wp(z + w2) - e2)/D2, everything integrates to
# Weierstrass zeta functions:
#   F1 = (c/2) (zeta(z) - zeta(z + w2))
#   F2 = (i c/2) (-zeta(z + w2) - zeta(z) - 2 e2 z),  F3 = z.

def _riemann_X(z, p, L):
    tau = p['tau']
    w2 = 0.5 * (1.0 + tau)
    e1 = L.wp(0.5)
    e2 = L.wp(w2)
    e3 = L.wp(0.5 * tau)
    D2 = (e2 - e1) * (e2 - e3)
    c = 1.0 / np.sqrt(D2)
    zw = L.zeta(z + w2)
    z0 = L.zeta(z)
    F1 = 0.5 * c * (z0 - zw)
    F2 = 0.5j * c * (-zw - z0 - 2.0 * e2 * z)
    return np.real(F1), np.real(F2), np.real(z)


# ==========================================================================
# WE_SURFACES: one row per Weierstrass surface
# ==========================================================================

def _knoid_phi(z, p):
    n = p['n']
    den = (z ** n - 1.0) ** 2
    zm = z ** (n - 1)
    return (0.5 * (1.0 - z ** (2 * (n - 1))) / den,
            0.5j * (1.0 + z ** (2 * (n - 1))) / den,
            zm / den)


def _tower_phi(z, p):
    n = p['n']
    den = z ** (2 * n) + 1.0
    return (0.5 * (1.0 - z ** (2 * (n - 1))) / den,
            0.5j * (1.0 + z ** (2 * (n - 1))) / den,
            z ** (n - 1) / den)


def _tower_roots(n):
    """The 2n ends of the saddle tower: the roots of z^{2n} = -1, all on
    the unit circle at the odd multiples of pi/(2n)."""
    return np.exp(1j * math.pi * (2 * np.arange(2 * n) + 1) / (2 * n))


def _tower_X(z, p, theta=0.0):
    """Exact antiderivative of the tower Weierstrass data (a sum of 2n
    logarithms, one per end), evaluated on the open unit disk.

    dh = z^{n-1}/(z^{2n}+1) dz and g = z^{n-1} give
        Int phi1 = -(1/4n) sum_rho (rho + 1/rho) ln(z - rho)
        Int phi2 = -(i/4n) sum_rho (rho - 1/rho) ln(z - rho)
        Int phi3 = -(1/2n) sum_rho  rho^n      ln(z - rho),
    the residue expansion of the rational integrand over its 2n simple
    poles rho (the ends).  Each logarithm's branch cut is placed radially
    *outward* from its end -- rho sits on the unit circle, so every cut
    lies outside the disk and the immersion is single-valued and smooth on
    it.  This closed form gives the true, unbounded planar (vertical-half-
    plane) ends: the numeric radial integrator instead caps and folds them
    into a torn blob, since rays from the origin only ever graze the poles.
    """
    n = p['n']
    rho = _tower_roots(n)
    d = np.asarray(z)[..., None] - rho
    dr = d * np.conj(rho)                       # rotate: outward ray -> +real
    L = np.log(np.abs(dr)) + 1j * np.mod(np.angle(dr), TAU)
    c1 = -(1.0 / (4 * n)) * (rho + 1.0 / rho)
    c2 = -(1j / (4 * n)) * (rho - 1.0 / rho)
    c3 = -(1.0 / (2 * n)) * (rho ** n)
    rot = np.exp(1j * theta)
    return (np.real(np.sum(c1 * L, axis=-1) * rot),
            np.real(np.sum(c2 * L, axis=-1) * rot),
            np.real(np.sum(c3 * L, axis=-1) * rot))


def _r_reach(radius):
    """Shared 'how far past the unit circle' domain radius mapping."""
    return 1.35 + 0.5 * min(max(radius / 1.2, 0.0), 1.6)


# --- generalized Enneper with a Bonnet (associate) angle ------------------
# Enneper of order k has g = z^k, dh = 2 z^k dz (a disk domain), so the
# Weierstrass 1-forms are entire polynomials -- every period vanishes
# identically and the associate family X = Re[e^{i theta} Int phi] is a
# smooth, tearing-free deformation of the base surface (theta = 0) into its
# conjugate (theta = pi/2).  The immersion is known in closed form, so it is
# meshed straight from the antiderivative (no radial quadrature): at theta=0
# this reproduces the classical Enneper parametrization exactly.

def _enneper_X(z, p, theta=0.0):
    k = p['k']
    rot = np.exp(1j * theta)
    zp = z ** (2 * k + 1) / (2 * k + 1)
    F1 = z - zp
    F2 = 1j * (z + zp)
    F3 = 2.0 * z ** (k + 1) / (k + 1)
    return (np.real(rot * F1), np.real(rot * F2), np.real(rot * F3))


# --- Enneper-ended k-noid --------------------------------------------------
# A genus-0 surface with n symmetric ends at the n-th roots of unity, each a
# higher-order (Enneper-type, winding) end rather than a Jorge-Meeks
# catenoid end.  The catenoid ends of the k-noid have a Gauss map that is
# finite and non-zero at the end; here the Gauss map g = (z^n - 1)/z instead
# *vanishes* at every end, so the horizontal coordinates grow like the square
# of the end coordinate (Enneper flaring) rather than logarithmically.  The
# height differential dh = (z^{n-1} + t z^{2n-1})/(z^n - 1)^3 dz has a triple
# pole at each end; regularity of the interior points z = 0 and z = infinity
# and the n-fold symmetry force this two-term numerator, and the single real
# weight t = t(n) is fixed by the one non-trivial period condition (only the
# phi2 real period does not already vanish by symmetry).  With that period
# closed, every 1-form residue is real, so the immersion has a closed form:
# a sum of logarithms (the residues) plus a rational partial-fraction part
# (the m >= 2 principal parts), evaluated directly on the disk -- no radial
# quadrature, hence clean flaring wings.  The two-term weight and the
# partial-fraction coefficients are extracted once per n by contour
# integration (cached) and the antiderivative is checked against its own
# derivative dX/dz = phi before use.  (n = 2 is the Double Enneper already in
# the catalog; this row covers n >= 3.)

_ENNK_CACHE = {}


def _ennk_g(z, p):
    n = p['n']
    return (z ** n - 1.0) / z


def _ennk_dh(z, p):
    n = p['n']
    return (z ** (n - 1) + p['t'] * z ** (2 * n - 1)) / (z ** n - 1.0) ** 3


def _ennk_phi(z, p):
    g = _ennk_g(z, p)
    dh = _ennk_dh(z, p)
    return (0.5 * (1.0 / g - g) * dh, 0.5j * (1.0 / g + g) * dh, dh)


def _ennk_solve(p):
    n = p['n']
    if n not in _ENNK_CACHE:
        # the phi2 real period is linear in the weight t = b/a; the other
        # two components already have vanishing real periods by symmetry
        def real_period(dh):
            f = (lambda z: 0.5j * (1.0 / _ennk_g(z, {'n': n})
                                   + _ennk_g(z, {'n': n})) * dh(z))
            return we.period_integral(f, 1.0, 0.06, 0.06).real
        P = real_period(lambda z: z ** (n - 1) / (z ** n - 1.0) ** 3)
        Q = real_period(lambda z: z ** (2 * n - 1) / (z ** n - 1.0) ** 3)
        t = -P / Q
        pn = {'n': n, 't': t}
        # partial-fraction coefficients c[j, end, m], m = 1..4, of each
        # 1-form component at each end (m = 1 is the residue -> a log term)
        rho = np.exp(2j * math.pi * np.arange(n) / n)
        C = np.zeros((3, n, 5), complex)
        for k, rk in enumerate(rho):
            for j in range(3):
                for m in range(1, 5):
                    C[j, k, m] = we.period_integral(
                        lambda z, j=j, rk=rk, m=m:
                        _ennk_phi(z, pn)[j] * (z - rk) ** (m - 1),
                        rk, 0.18, 0.18) / (TAU * 1j)
        _ENNK_CACHE[n] = (t, rho, C)
    t, rho, C = _ENNK_CACHE[n]
    return dict(p, t=t, _rho=rho, _C=C)


def _ennk_X(z, p, theta=0.0):
    """Closed-form immersion: sum of end logarithms + rational principal
    parts.  Each end's log branch cut is rotated to point radially outward
    from the unit circle, so no cut crosses the meshed disk."""
    rho, C = p['_rho'], p['_C']
    z = np.asarray(z, complex)
    out = []
    for j in range(3):
        s = np.zeros(z.shape, complex)
        for k in range(len(rho)):
            d = z - rho[k]
            dr = d * np.conj(rho[k])              # outward ray -> +real axis
            logd = np.log(np.abs(dr)) + 1j * np.angle(dr) + np.log(rho[k])
            s = s + C[j, k, 1] * logd
            for m in range(2, 5):
                s = s + C[j, k, m] * (-1.0 / ((m - 1) * d ** (m - 1)))
        out.append(np.real(s))
    return out[0], out[1], out[2]


WE_SURFACES = {
    # --- ports of the former bespoke toolkit builders ---------------------
    'KNOID': {
        'label': "Jorge-Meeks k-noid",
        'family': 'SPHERES',
        'phi': _knoid_phi,
        'domain': ('disk', 0.0, lambda p: p['r_out']),
        'p_from': lambda order, radius: {
            'n': int(max(3, min(order, 12))), 'r_out': _r_reach(radius)},
        'count': "Ends (n)",
        'clip': True,
        # the catenoid ends flare out at the n-th roots of unity; a denser
        # grid lets the object-space end trim + boundary relaxation leave a
        # clean rounded rim instead of the coarse ragged one the 48x48
        # grid tore
        'res_boost': (2.2, 2.2),
        'cycles': lambda p: [(np.exp(2j * math.pi * j / p['n']), 0.18)
                             for j in range(p['n'])],
        'test_order': 4,
    },
    'COSTA_HM': {
        'label': "Costa-Hoffman-Meeks (genus k)",
        'family': 'TORI',
        'phi': _chm_phi,
        'domain': ('halfplane', 12.0),
        'p_from': lambda order, radius: {
            'k': int(max(1, min(order, 6)))},
        'solve': lambda p: dict(p, c=chm_modulus(p['k'])),
        'symmetry': {'rot': lambda p: p['k'] + 1, 'mirror': True},
        'r_grade': lambda p: (p['k'] + 1) / 2,
        'seed': _chm_seed,
        'hp_punctures': lambda p: [(1.0, 0.09), (-1.0, 0.09)],
        'count': "Genus (k)",
        'test_order': 2,
    },
    # --- Tier 0/1: new surfaces from textbook Weierstrass data ------------
    'SCHERK_TOWER': {
        # Karcher's most symmetric saddle tower; n = 2 is Scherk's
        # singly periodic surface (the conjugate of SCHERK1).  Built from
        # the exact log-sum antiderivative (_tower_X) over the open unit
        # disk: the 2n ends sit on the unit circle, and a small circular
        # puncture around each maps to a clean, near-straight wing rim
        # (the ends are logarithmic/planar, so their conformal coordinate
        # is log).  The former radial disk integrator capped and folded
        # the (infinite) wings into a torn blob and is not used here.
        'label': "Saddle Tower (Scherk singly periodic)",
        'family': 'SINGLY',
        'phi': _tower_phi,           # kept for the period-closure gate
        'Xexact': _tower_X,          # exact immersion used by the mesher
        'domain': ('disk', 0.0, 0.999),
        # n = wings-per-turn; the UI "order" 1..7 maps to n = 2..8 so the
        # slider's first step already changes the surface (n must be >= 2:
        # n = 2 is the classical 4-wing Scherk saddle tower).  The radius
        # slider shrinks the end punctures, so the wings reach further.
        'p_from': lambda order, radius: {
            'n': int(min(max(order + 1, 2), 8)),
            'eps': 0.085 / min(max(radius / 1.2, 0.5), 2.0)},
        'count': "Wings (n pairs)",
        'mask_punctures': lambda p: [
            (rho, p['eps']) for rho in _tower_roots(p['n'])],
        'clip': False,
        'res_boost': (2.6, 2.6),     # dense grid -> clean puncture rims
        'cycles': lambda p: [
            (np.exp(1j * math.pi * (2 * j + 1) / (2 * p['n'])), 0.12)
            for j in range(2 * p['n'])],
        'cycle_free': (2,),          # vertical translation is the period
        'test_order': 3,
    },
    'RICHMOND_K': {
        # generalized Richmond: planar end + higher-order flower,
        # g = z^k, dh = dz; k = 2 is the classical Richmond surface
        'label': "Richmond (generalized, g = z^k)",
        'family': 'CLASSICAL',
        'g': lambda z, p: z ** p['k'],
        'dh': lambda z, p: np.ones_like(z),
        'domain': ('disk', 0.25, lambda p: 0.25 + p['reach']),
        'p_from': lambda order, radius: {
            'k': int(min(max(order + 1, 2), 12)), 'reach': radius},
        'count': "Order (k)",
        'clip': False,
        'radial_grade': 'rim',       # planar end grows toward r_out
        'res_boost': (1.7, 1.5),
        'cycles': lambda p: [(0.0, 0.5)],
        # k >= 2 has no residue at the planar end, so every period stays zero
        # under phi *= e^{i theta}: the associate/Bonnet deformation is
        # single-valued and tear-free (the annulus base ring closes for all
        # theta).  theta = 0 is Richmond; sweeping it rotates the flower.
        'associate': True,
        'test_order': 2,
    },
    'DOUBLE_ENNEPER': {
        # two Enneper ends (z = 0 and z = infinity) joined in the
        # middle; period closure at the a = -1 coefficient (residue
        # calc in the zoo tests): g = z, dh = (z^2 - 1 + z^-2) dz
        'label': "Double Enneper",
        'family': 'SPHERES',
        'g': lambda z, p: z,
        'dh': lambda z, p: z * z - 1.0 + 1.0 / (z * z),
        'domain': ('disk', lambda p: 1.0 / p['r1'], lambda p: p['r1']),
        'p_from': lambda order, radius: {
            'r1': 1.8 + 0.8 * min(max(radius / 1.2, 0.0), 2.0)},
        'clip': True,
        'radial_grade': 'both',      # Enneper ends at r_in and r_out
        'res_boost': (1.7, 2.3),
        'cycles': lambda p: [(0.0, 1.0)],
        'test_order': 1,
    },
    'MEEKS_MOBIUS': {
        # Meeks' complete minimal Mobius strip (Duke Math. J. 1981):
        # g = z^2 (z+1)/(z-1), dh = i (z^2-1)/z^2 dz on C - {0}; the
        # annulus double-covers the strip (z ~ -1/conj(z))
        'label': "Meeks Mobius Strip",
        'family': 'NONORIENT',
        'phi': lambda z, p: (
            0.5j * ((z - 1.0) ** 2 / z ** 4 - (z + 1.0) ** 2),
            -0.5 * ((z - 1.0) ** 2 / z ** 4 + (z + 1.0) ** 2),
            1j * (z * z - 1.0) / (z * z)),
        'domain': ('disk', lambda p: 1.0 / p['r1'], lambda p: p['r1']),
        'p_from': lambda order, radius: {
            'r1': 1.6 + 0.6 * min(max(radius / 1.2, 0.0), 2.0)},
        'clip': True,
        'res_boost': (2.6, 2.6),     # annulus double-cover reads ragged at 48
        'cycles': lambda p: [(0.0, 1.0)],
        'test_order': 1,
    },
    'RIEMANN': {
        # Riemann's singly periodic minimal example (1867), foliated
        # by circles in horizontal planes; two copies of the
        # translational fundamental domain, cut at the planar ends
        'label': "Riemann's Minimal Example",
        'family': 'SINGLY',
        'X': _riemann_X,
        'domain': ('torus', 0.5, lambda p: p['tau']),
        'p_from': lambda order, radius: {
            'tau': 1j * min(max(0.5 + 0.5 * radius / 1.2, 0.7), 2.2)},
        'punctures': lambda p: [(0.0, 0.0, 0.17), (0.5, 0.5, 0.17)],
        'torus_wrap': (False, True),
        'copies': 2,
        'test_order': 1,
    },
    # --- Tier 0 (M2): associate-angle surfaces and Enneper-ended k-noid ----
    # NB: appended in a self-contained block; the saddle-tower row above is
    # owned by a separate work stream -- do not fold these into it.
    'ENNEPER': {
        # generalized Enneper (order k) with a Bonnet associate slider,
        # rebuilt on the WE engine so theta sweeps Enneper <-> its conjugate.
        # Overrides the toolkit's closed-form Enneper; identical at theta = 0.
        'label': "Enneper",
        'family': 'CLASSICAL',
        'g': lambda z, p: z ** p['k'],          # for the period-closure gate
        'dh': lambda z, p: 2.0 * z ** p['k'],
        'Xexact': _enneper_X,                    # exact immersion (+ theta)
        'domain': ('disk', 0.0, lambda p: p['reach']),
        'p_from': lambda order, radius: {
            'k': int(min(max(order, 1), 12)), 'reach': radius},
        'count': "Enneper order (k)",
        'associate': True,
        'clip': False,
        'cycles': lambda p: [(0.0, 0.5)],        # entire phi -> zero periods
        'test_order': 1,
    },
    'ENNK': {
        # Enneper-ended n-noid: genus 0, n winding (Enneper) ends at the
        # n-th roots of unity.  g = (z^n-1)/z, dh = (z^{n-1}+t z^{2n-1})
        # /(z^n-1)^3, t = t(n) closing the one non-trivial period.
        'label': "Enneper-ended k-noid",
        'family': 'SPHERES',
        'phi': _ennk_phi,                        # for the period-closure gate
        'Xexact': _ennk_X,                       # exact immersion (log+rat'l)
        'domain': ('disk', 0.0, 0.985),
        'p_from': lambda order, radius: {'n': int(max(3, min(order, 7)))},
        'solve': _ennk_solve,
        'count': "Ends (n)",
        # end punctures give clean flaring-wing rims; grow eps with n so the
        # tighter high-n wings stay tear-free
        'mask_punctures': lambda p: [
            (r, 0.14 + 0.02 * p['n']) for r in p['_rho']],
        'radial_grade': 'rim',                   # cluster nodes near the ends
        'clip': False,
        'res_boost': (3.2, 4.2),                 # dense rims, smooth wings
        'cycles': lambda p: [(r, 0.12) for r in p['_rho']],
        'test_order': 1,                         # order 1 -> n = 3 (trinoid)
    },
}


# ==========================================================================
# BJORLING: curve-seeded strips (Schwarz's formula)
# ==========================================================================

def _bj_circle_normal(w, p):
    m = p['m']
    return (np.cos(0.5 * m * w) * np.cos(w),
            np.cos(0.5 * m * w) * np.sin(w),
            np.sin(0.5 * m * w))


# Bjorling seed curves that wind quickly (spirals, the trefoil) still read
# a little polygonal on the shared 48x48 grid once the strip is correctly
# indexed, so give them a modest denser grid along the curve direction --
# where the curvature lives -- for a smoother ribbon by default.
_BJ_BOOST = (2.0, 1.2)


BJORLING = {
    'BJ_CYCLOID': {
        # free regression check: the cycloid seed with its principal
        # normal reproduces Catalan's surface exactly
        'label': "Bjorling: Cycloid (Catalan)",
        'family': 'BJORLING',
        'curve': lambda w, p: (w - np.sin(w), 1.0 - np.cos(w), 0.0 * w),
        'normal': lambda w, p: (np.cos(w / 2), -np.sin(w / 2), 0.0 * w),
        't_range': (-math.pi, 3 * math.pi),
        'v_half': lambda p: p['vh'],
        'p_from': lambda order, radius: {'vh': min(radius, 2.2)},
        'res_boost': _BJ_BOOST,
        'associate': True,
    },
    'BJ_CIRCLE': {
        # circle seed with a normal that makes m half-twists along the
        # loop: m = 1 is a minimal Mobius band
        # TODO(zoo): weld the u-seam for even m (closed orientable band)
        'label': "Bjorling: Twisted Band (Mobius)",
        'family': 'BJORLING',
        'curve': lambda w, p: (np.cos(w), np.sin(w), 0.0 * w),
        'normal': _bj_circle_normal,
        't_range': (0.0, TAU),
        'v_half': lambda p: p['vh'],
        'p_from': lambda order, radius: {
            'm': int(min(max(order, 1), 7)),
            'vh': 0.35 * min(max(radius / 1.2, 0.3), 2.0)},
        'res_boost': _BJ_BOOST,
        'count': "Half-twists",
        'associate': True,
    },
    'BJ_HELIX': {
        'label': "Bjorling: Helix Ribbon",
        'family': 'BJORLING',
        'curve': lambda w, p: (np.cos(w), np.sin(w), 0.4 * w),
        'normal': lambda w, p: (-np.cos(w), -np.sin(w), 0.0 * w),
        't_range': (-TAU, TAU),
        'v_half': lambda p: p['vh'],
        'p_from': lambda order, radius: {
            'vh': 0.8 * min(max(radius / 1.2, 0.25), 2.0)},
        'res_boost': _BJ_BOOST,
        'associate': True,
    },
    'BJ_TREFOIL': {
        'label': "Bjorling: Trefoil Ribbon",
        'family': 'BJORLING',
        'curve': lambda w, p: (
            0.5 * (2.0 + np.cos(3.0 * w)) * np.cos(2.0 * w),
            0.5 * (2.0 + np.cos(3.0 * w)) * np.sin(2.0 * w),
            0.5 * np.sin(3.0 * w)),
        'normal': 'frenet',
        't_range': (0.0, TAU),
        'closed': True,
        'v_half': lambda p: p['vh'],
        'p_from': lambda order, radius: {
            'vh': 0.22 * min(max(radius / 1.2, 0.25), 1.6)},
        'res_boost': _BJ_BOOST,
        'associate': True,
    },
    'BJ_ARCH_SPIRAL': {
        'label': "Bjorling: Archimedean Spiral",
        'family': 'BJORLING',
        'curve': lambda w, p: (0.25 * w * np.cos(w),
                               0.25 * w * np.sin(w), 0.0 * w),
        'normal': 'frenet',
        't_range': (0.6 * math.pi, 3.5 * math.pi),
        'v_half': lambda p: p['vh'],
        'p_from': lambda order, radius: {
            'vh': 0.8 * min(max(radius / 1.2, 0.25), 2.0)},
        'res_boost': _BJ_BOOST,
        'associate': True,
    },
    'BJ_LOG_SPIRAL': {
        'label': "Bjorling: Logarithmic Spiral",
        'family': 'BJORLING',
        'curve': lambda w, p: (
            0.25 * np.exp(0.15 * w) * np.cos(w),
            0.25 * np.exp(0.15 * w) * np.sin(w), 0.0 * w),
        'normal': 'frenet',
        't_range': (0.0, 4.0 * math.pi),
        'v_half': lambda p: p['vh'],
        'p_from': lambda order, radius: {
            'vh': 0.5 * min(max(radius / 1.2, 0.25), 2.0)},
        'res_boost': _BJ_BOOST,
        'associate': True,
    },
}

# TODO(zoo) -- deferred to the next batch (data not yet verified against
# Weber's notebooks; deliberately NOT registered rather than mislabeled):
#   * Pyramidal / bipyramidal / prismatic k-noids (Lopez-Ros parameter;
#     needs the degree-(n+1) Gauss map data + solve_scalar closure)
#   * Lopez spheres
#   * Bjorling clothoid (needs a complex Fresnel evaluator)
#   * Karcher's less-symmetric saddle towers (angle parameter alpha)
#   * genus-1 helicoid, Callahan-Hoffman-Meeks, KMR (Tier 3+)
#   * Bonnet angle on Henneberg (non-orientable -> the associate family is
#     not globally single-valued) and Bour (fractional-power double cover);
#     both stay as the toolkit's fixed closed forms.  Catalan's associate is
#     already reachable through BJ_CYCLOID.


SURFACE_FAMILY = dict(LEGACY_FAMILY)
for _k, _s in WE_SURFACES.items():
    SURFACE_FAMILY[_k] = _s['family']
for _k, _s in BJORLING.items():
    SURFACE_FAMILY[_k] = _s['family']


def register(PARAMETRIC=None, MESH_PARAM=None, COUNT_PARAM=None,
             ANGLE_PARAM=None):
    """Wire every catalog row into the toolkit's registries.  Called
    (with the four dicts) from minimal_surface_toolkit at import time;
    a bare register() call -- e.g. from the extension loader -- is a
    no-op, since this module has no Blender UI of its own."""
    if PARAMETRIC is None:
        return
    for key, spec in WE_SURFACES.items():
        b = we.make_entry(key, spec)
        PARAMETRIC[key] = (spec['label'], b)
        if b.finished_mesh:
            MESH_PARAM[key] = b
        if spec.get('count'):
            COUNT_PARAM[key] = spec['count']
        if spec.get('associate'):
            ANGLE_PARAM.add(key)
    for key, spec in BJORLING.items():
        b = we.make_bjorling_entry(key, spec)
        PARAMETRIC[key] = (spec['label'], b)
        if spec.get('count'):
            COUNT_PARAM[key] = spec['count']
        if spec.get('associate'):
            ANGLE_PARAM.add(key)


ADD_MENU = True


def unregister():
    pass


if __name__ == "__main__":
    # standalone catalog tests: build every row through the toolkit
    # pipeline, then the engine-level QA gates (period closure,
    # translation structure, Bjorling seed reproduction)
    import minimal_surface_toolkit as tk
    ok = True
    for key in list(WE_SURFACES) + list(BJORLING):
        spec = WE_SURFACES.get(key) or BJORLING[key]
        n = spec.get('test_order', 1)
        # exercise the associate/Bonnet angle on every surface that exposes
        # it, so a broken morph (tear / self-collapse) trips the fit or
        # manifold gate here rather than in Blender
        th = math.pi / 4 if spec.get('associate') else 0.0
        V, Q = tk.build_parametric(key, 60, 60, n, 1.2, 1.0, th)
        finite = bool(np.all(np.isfinite(V)))
        lo, hi = V.min(0), V.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        ec = {}
        for f in Q:
            m = len(f)
            for t in range(m):
                a, b = f[t], f[(t + 1) % m]
                e = (a, b) if a < b else (b, a)
                ec[e] = ec.get(e, 0) + 1
        nonman = sum(1 for c in ec.values() if c > 2)
        good = (finite and len(Q) > 100 and cen < 1e-6
                and abs(ext - 2.0) < 1e-6 and nonman == 0)
        ok &= good
        print(f"zoo {key:15s}: {len(V):5d} verts {len(Q):5d} faces "
              f"fit[|c|={cen:.1e} ext={ext:.4f}] nonman={nonman} "
              f"{'OK' if good else 'FAIL'}")
    # period-closure gates: Re of the contour integral of every phi
    # component vanishes around each listed cycle (up to the allowed
    # translation components)
    for key, spec in WE_SURFACES.items():
        if 'cycles' not in spec:
            continue
        p = spec['p_from'](spec.get('test_order', 1), 1.2)
        if 'solve' in spec:
            p = spec['solve'](p) or p
        phi = we._phi_fn(spec, p, 0.0)
        free = set(spec.get('cycle_free', ()))
        worst = 0.0
        for (zc, r) in (spec['cycles'](p) if callable(spec['cycles'])
                        else spec['cycles']):
            for comp in range(3):
                if comp in free:
                    continue
                I = we.period_integral(
                    lambda z, c=comp: phi(z)[..., c], zc, r, r)
                worst = max(worst, abs(I.real))
        good = worst < 1e-6
        ok &= good
        print(f"periods {key:15s}: max|Re oint phi| = {worst:.2e} "
              f"{'OK' if good else 'FAIL'}")
    # Riemann: the u -> u+1 deck map must be a constant translation
    # (0, y, 1) -- singly periodic, tilted, no x-component
    spec = WE_SURFACES['RIEMANN']
    p = spec['p_from'](1, 1.2)
    nu, nv = 121, 60
    x, y, z, wu, wv, mask = we.we_surface(spec, nu, nv, 1, 1.2)
    h = (nu - 1) // 2                      # one period = half the grid
    m2 = mask[:nu - h] & mask[h:]
    dx = np.where(m2, x[h:] - x[:nu - h], np.nan)
    dy = np.where(m2, y[h:] - y[:nu - h], np.nan)
    dz = np.where(m2, z[h:] - z[:nu - h], np.nan)
    sx = float(np.nanstd(dx)) + float(np.nanstd(dy)) + float(np.nanstd(dz))
    tx = abs(float(np.nanmean(dx)))
    tz = abs(float(np.nanmean(dz)) - 1.0)
    good = sx < 1e-6 and tx < 1e-6 and tz < 1e-6
    ok &= good
    print(f"Riemann translation: T=({np.nanmean(dx):.2e},"
          f"{np.nanmean(dy):.4f},{np.nanmean(dz):.4f}) spread={sx:.2e} "
          f"{'OK' if good else 'FAIL'}")
    # Bjorling seed gate: the v = 0 row of every strip reproduces its
    # seed curve to machine precision
    for key, spec in BJORLING.items():
        p = spec['p_from'](spec.get('test_order', 1), 1.2)
        x, y, z, wu, _, _ = we.bjorling_surface(spec, 90, 61, 1, 1.2)
        t0, t1 = spec['t_range']
        u = np.linspace(t0, t1, 90, endpoint=not wu)
        cx, cy, cz = spec['curve'](u.astype(complex), p)
        jm = 61 // 2
        err = max(np.max(np.abs(x[:, jm] - np.real(cx))),
                  np.max(np.abs(y[:, jm] - np.real(cy))),
                  np.max(np.abs(z[:, jm] - np.real(cz))))
        good = err < 1e-9
        ok &= good
        print(f"seed {key:15s}: err={err:.2e} {'OK' if good else 'FAIL'}")
    # associate/Bonnet morph gate: theta = 0 reproduces the base surface and
    # the deformation is continuous (a small step gives a bounded, non-torn
    # change).  Checked on the closed-form engine associates on a fixed grid
    # (endpoint-free wraps drop at theta > 0, so compare interior columns).
    for key in ('ENNEPER', 'RICHMOND_K'):
        spec = WE_SURFACES[key]
        p0 = spec['p_from'](spec.get('test_order', 1), 1.2)
        n0 = spec.get('test_order', 1)

        def raw(th):
            x, y, z, _, _, _ = we.we_surface(spec, 80, 80, n0, 1.2, None, th)
            return np.stack([x, y, z], axis=-1)
        base = raw(0.0)
        d1 = float(np.nanmax(np.abs(raw(0.02) - base)))
        d2 = float(np.nanmax(np.abs(raw(0.04) - base)))
        # continuous: a 2x larger angle step gives a ~2x (bounded) change,
        # never a blow-up; and the morph actually moves (non-degenerate)
        cont = (np.isfinite(d1) and np.isfinite(d2)
                and 1e-4 < d1 < 1.0 and d2 < 3.0 * d1 + 1e-6)
        thmid = raw(math.pi / 4)
        moved = float(np.nanmax(np.abs(thmid - base))) > 0.05
        good = cont and moved
        ok &= good
        print(f"assoc {key:13s}: d(.02)={d1:.2e} d(.04)={d2:.2e} "
              f"moved={moved} {'OK' if good else 'FAIL'}")
    # CHM modulus regression (must match the pre-port table)
    cref = {1: 0.955978, 2: 0.988070, 3: 0.995117, 4: 0.997535}
    cok = all(abs(chm_modulus(kk) - cref[kk]) < 1e-5 for kk in cref)
    ok &= cok
    print("CHM modulus: "
          + " ".join(f"c({kk})={chm_modulus(kk):.6f}" for kk in cref)
          + f"  {'OK' if cok else 'FAIL'}")
    print("\nRESULT:", "ALL OK" if ok else "FAILURES in zoo")
