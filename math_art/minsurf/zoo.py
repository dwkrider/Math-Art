
# Minimal-surface zoo: the data catalog for the Weierstrass-Enneper /
# Bjorling engine in weierstrass.py.
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

from . import weierstrass as we

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


# --- Karcher's less-symmetric saddle tower (the alpha family) -------------
# The symmetric saddle tower (SCHERK_TOWER, above) puts its 2n wing ends at
# the equally spaced 2n-th roots of z^{2n} = -1.  Karcher's generalization
# ("Embedded minimal surfaces derived from Scherk's examples", Manuscripta
# Math. 62, 1988) lets the ends *cluster into pairs*: the 2n ends sit in n
# symmetric pairs about the directions 2 pi k / n, each pair split by a
# half-angle gamma.  gamma = pi/(2n) is exactly the equally spaced symmetric
# tower; shrinking gamma opens two-and-narrows-two of the wing walls -- the
# unequal-wing saddle.  The Weierstrass data keeps the same *form* as the
# symmetric case, g = z^{n-1} and dh = z^{n-1}/D(z) dz with D(z) = prod
# (z - rho_j) over the 2n chosen ends rho_j on the unit circle, i.e. a
# rational height differential with a simple pole at each end.
#
# Period closure.  Moving the ends off the equally spaced positions would in
# general open the horizontal (real) periods and tear the wings apart.  Here
# the ends are placed as a *symmetric* configuration -- invariant under the
# real- and imaginary-axis reflections and the 2 pi/n rotation (the dihedral
# group D_n) -- and g = z^{n-1} respects that symmetry.  That is enough to
# force every phi1, phi2 residue at every end to be real, so Re oint phi_h
# vanishes automatically at every end for *all* gamma (verified to < 5e-15
# by contour integration over n = 2..6 across the whole alpha range in the
# standalone tests): no residual weight has to be solved for.  Only phi3 = dh
# keeps a nonzero real period +-V(gamma) -- the vertical translation that
# makes the surface singly periodic.  The immersion is then the same exact
# log-sum antiderivative as _tower_X, with the residues (c1, c2, c3) read off
# each end via 1/D'(rho); at gamma = pi/(2n) it reduces term-by-term to
# _tower_X, so alpha = 0 reproduces the symmetric unit exactly.
#
# One fundamental domain (one vertical period) is meshed.  Unlike the
# equally spaced tower, the unequal-wing unit admits no screw deck isometry
# that welds one disk unit onto the next (its only rigid symmetries are the
# internal reflections/rotation, all with zero vertical shift), so multiple
# storeys are not stacked here -- see the TODO(zoo) note.  The symmetric
# tower (SCHERK_TOWER) remains the stackable alpha = 0 special case.

def _atower_ends(n, alpha):
    """The 2n ends: n symmetric pairs about the directions 2 pi k / n, each
    split by +-gamma(alpha).  alpha in [0, pi/2]; alpha = 0 -> gamma = pi/(2n)
    (the equally spaced symmetric tower), larger alpha closes the pairs."""
    gsym = math.pi / (2 * n)
    frac = 1.0 - 0.85 * (alpha / (math.pi / 2.0))
    gamma = gsym * min(max(frac, 0.05), 1.0)
    k = np.arange(n)
    a = np.concatenate([k * TAU / n + gamma, k * TAU / n - gamma])
    return np.exp(1j * a)


def _atower_coeffs(n, alpha):
    """(ends, c1, c2, c3): the residue of each phi component at each end, from
    the partial-fraction 1/D'(rho) of the rational Weierstrass forms
    phi1 = (1-z^{2(n-1)})/2D, phi2 = i(1+z^{2(n-1)})/2D, phi3 = z^{n-1}/D."""
    rho = _atower_ends(n, alpha)
    Dp = np.array([np.prod([rho[j] - rho[k]
                            for k in range(len(rho)) if k != j])
                   for j in range(len(rho))])
    c1 = 0.5 * (1.0 - rho ** (2 * (n - 1))) / Dp
    c2 = 0.5j * (1.0 + rho ** (2 * (n - 1))) / Dp
    c3 = rho ** (n - 1) / Dp
    return rho, c1, c2, c3


def _atower_phi(z, p):
    """Weierstrass 1-forms of the alpha tower (kept for the period-closure
    gate): D(z) = prod (z - end) built directly from the end positions."""
    n = p['n']
    z = np.asarray(z, complex)
    rho = _atower_ends(n, p.get('alpha', 0.0))
    D = np.ones_like(z)
    for r in rho:
        D = D * (z - r)
    return (0.5 * (1.0 - z ** (2 * (n - 1))) / D,
            0.5j * (1.0 + z ** (2 * (n - 1))) / D,
            z ** (n - 1) / D)


def _atower_X(z, p, theta=0.0):
    """Exact log-sum immersion of the alpha tower on the punctured disk --
    the same closed form as _tower_X (a sum of 2n end logarithms, one per
    residue, each with its branch cut rotated radially outward so no cut
    crosses the disk), but with the alpha-controlled end positions and their
    general 1/D'(rho) residues.  Reduces to _tower_X at alpha = 0."""
    n = p['n']
    rho, c1, c2, c3 = _atower_coeffs(n, p.get('alpha', 0.0))
    d = np.asarray(z)[..., None] - rho
    dr = d * np.conj(rho)                       # rotate: outward ray -> +real
    L = np.log(np.abs(dr)) + 1j * np.mod(np.angle(dr), TAU)
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


# --- M3 batch helpers: higher genus (Chen-Gackstatter) + symmetrizations
# (harvested from minimalsurfaces.blog) -----------------------------------
# Two clean, fully verified additions land in the WE_SURFACES block at the
# end of this dict: the Tilted Scherk (a Lopez-Ros deformation of the
# doubly periodic Scherk surface) and Enneper-with-n-catenoids (a
# symmetrized genus-0 surface with one Enneper end and n catenoidal ends).
# References:
#   R. Schoen / Lopez & Ros (1991) on the Lopez-Ros deformation; H. F.
#   Scherk (1835) for the base Scherk surface; the decorated-Enneper
#   symmetrization follows M. Weber's repository,
#   https://minimalsurfaces.blog/ (Enneper with n Catenoids).

def _tiltscherk_roots():
    """The four ends of the tilted Scherk surface: the 4th roots of unity
    {1, i, -1, -i} (the poles of dh = 4z/(z^4 - 1))."""
    return np.exp(1j * math.pi * np.arange(4) / 2.0)


def _tiltscherk_phi(z, p):
    """Weierstrass 1-forms of the tilted Scherk surface (kept for the
    period-closure gate): g = Rho z, dh = 4z/(z^4 - 1)."""
    Rho = p['Rho']
    d = z ** 4 - 1.0
    return (2.0 * (1.0 / Rho - Rho * z * z) / d,
            2.0j * (1.0 / Rho + Rho * z * z) / d,
            4.0 * z / d)


def _tiltscherk_X(z, p, theta=0.0):
    """Exact log-sum immersion of the tilted Scherk surface on the open
    unit disk.  dh = 4z/(z^4 - 1) has a simple pole at each 4th root of
    unity rho, so the antiderivative is a sum of four logarithms with
    residues read off 1/(4 rho^3) = rho/4; each log's branch cut is rotated
    radially outward (rho is on the unit circle) so no cut crosses the
    disk and the immersion is single-valued and smooth on it.  The Lopez-Ros
    parameter Rho tilts the four ends (the doubly periodic 'tilt'); Rho = 1
    is the symmetric Scherk saddle.  Two of the four end periods are the
    genuine doubly periodic lattice translations (in x at z = +-i, in y at
    z = +-1); only the vertical period vanishes, so this single fundamental
    saddle is meshed straight from the antiderivative (no radial quadrature,
    hence clean planar wing ends)."""
    Rho = p['Rho']
    rho = _tiltscherk_roots()
    c1 = (rho / 2.0) * (1.0 / Rho - Rho * rho ** 2)
    c2 = (1j * rho / 2.0) * (1.0 / Rho + Rho * rho ** 2)
    c3 = rho ** 2
    d = np.asarray(z)[..., None] - rho
    dr = d * np.conj(rho)                       # rotate: outward ray -> +real
    L = np.log(np.abs(dr)) + 1j * np.mod(np.angle(dr), TAU)
    rot = np.exp(1j * theta)
    return (np.real(np.sum(c1 * L, axis=-1) * rot),
            np.real(np.sum(c2 * L, axis=-1) * rot),
            np.real(np.sum(c3 * L, axis=-1) * rot))


# Enneper-with-n-catenoids (decorated Enneper): genus 0, one Enneper end at
# z = infinity plus n catenoidal ends at the n-th roots of unity (the double
# poles of dh).  The period problem is solved in closed form via residues:
# with the free shape parameter a fixed, the Lopez-Ros factor is identically
# 1 and the neck radius is b = (2 - a^n)^(1/n) (real for a^n < 2), so no
# numeric solve is needed -- every real period vanishes to machine epsilon
# (verified by contour integration for n = 2..4).  g = z^{n-1}(z^n - b^n)/
# (z^n - a^n), dh = z^{n-1}(z^n - a^n)(z^n - b^n)/(z^n - 1)^2 dz.

def _enc_roots(n):
    """The n catenoidal ends: the n-th roots of unity (the double poles of
    the height differential)."""
    return np.exp(2j * math.pi * np.arange(n) / n)


def _enc_b(n, a):
    """Closed-form neck radius b = (2 - a^n)^(1/n) that closes the period
    problem for the given free shape parameter a (needs a^n < 2)."""
    return (2.0 - a ** n) ** (1.0 / n)


def _enc_phi(z, p):
    n, a, b = p['n'], p['a'], p['b']
    zn = z ** n
    g = z ** (n - 1) * (zn - b ** n) / (zn - a ** n)
    dh = z ** (n - 1) * (zn - a ** n) * (zn - b ** n) / (zn - 1.0) ** 2
    return (0.5 * (1.0 / g - g) * dh, 0.5j * (1.0 / g + g) * dh, dh)


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
        # clean rounded rim instead of the coarse ragged one a sparse grid
        # tore.  Rebalanced for the 64 res baseline (was 2.2 at 48).
        'res_boost': (1.8, 1.8),
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
        # singly periodic surface (the conjugate of SCHERK1).  One
        # fundamental domain is a single 2n-winged saddle from the exact
        # log-sum antiderivative (_tower_X) over the open unit disk: the
        # 2n ends sit on the unit circle, and a small circular puncture
        # around each maps to a clean, near-straight wing rim (the ends
        # are logarithmic/planar, so their conformal coordinate is log).
        #
        # A saddle tower is periodic under a vertical *screw motion*: dh =
        # phi3 has residue +-i/(2n) at each end, so the vertical monodromy
        # around one end is Re(2 pi i * residue) = +-pi/n; the disk sees
        # only a half-turn about each logarithmic end, so one fundamental
        # domain spans T = pi/(2n) in height, and the deck isometry that
        # stacks the next storey translates by T and rotates by pi/n about
        # the axis (the rotation permutes the 2n wings, so the vertical
        # wing-walls join storey-to-storey into continuous planes).  The
        # mesher (we_saddle_tower) stacks `storeys` copies under that screw
        # and welds the shared wing rims, so the default renders as a
        # genuinely repeating tower rather than a lone saddle.
        'label': "Saddle Tower (Scherk singly periodic)",
        'family': 'SINGLY',
        'phi': _tower_phi,           # kept for the period-closure gate
        'Xexact': _tower_X,          # exact immersion used by the mesher
        'tower': True,               # -> we_saddle_tower finished mesh
        'domain': ('disk', 0.0, 0.999),
        # n = wings-per-turn; the UI "Wings" (order) 1..7 maps to n = 2..8
        # so the slider's first step already changes the surface (n must be
        # >= 2: n = 2 is the classical 4-wing Scherk saddle tower).  The
        # radius slider shrinks the end punctures, so the wings reach
        # further.  "Storeys" is a separate count (how many stacked units).
        'p_from': lambda order, radius: {
            'n': int(min(max(order + 1, 2), 8)),
            'eps': 0.085 / min(max(radius / 1.2, 0.5), 2.0)},
        'count': "Wings (n pairs)",
        'storeys_label': "Storeys",
        'mask_punctures': lambda p: [
            (rho, p['eps']) for rho in _tower_roots(p['n'])],
        'clip_punctures': True,       # snap wing rims onto the mask circle
        'clip': False,
        # the 2n ends sit on |z| = 1; grading the radial samples toward the
        # rim puts the fine cells exactly where the punctures cut, so each
        # wing rim starts as a tiny-step staircase the boundary relaxation
        # then finishes smooth (instead of the coarse one-quad scallop).
        # With the grading carrying the rim, a modest raw density suffices
        # (keeps the 3-storey default tower under ~28k verts).
        'radial_grade': 'rim',
        'res_boost': (1.5, 1.5),
        'cycles': lambda p: [
            (np.exp(1j * math.pi * (2 * j + 1) / (2 * p['n'])), 0.12)
            for j in range(2 * p['n'])],
        'cycle_free': (2,),          # vertical translation is the period
        'test_order': 1,             # n = 2, the classical 4-wing tower
    },
    'SADDLE_TOWER_A': {
        # Karcher's less-symmetric saddle tower (the alpha family): the 2n
        # wing ends cluster into n symmetric pairs, so the walls become
        # wide/narrow openings.  alpha (exposed on the associate-angle knob)
        # runs 0 -> pi/2; alpha = 0 is the equally spaced symmetric unit
        # (identical to SCHERK_TOWER's fundamental domain).  Period closure
        # is automatic from the dihedral end placement (see the header block
        # above); one fundamental domain -- one full vertical period -- is
        # meshed, from the same exact log-sum immersion (_atower_X).
        'label': "Saddle Tower (Karcher, unequal wings)",
        'family': 'SINGLY',
        'phi': _atower_phi,          # kept for the period-closure gate
        'Xexact': _atower_X,         # exact immersion used by the mesher
        'tower': True,               # -> we_saddle_tower (single unit)
        'alpha_from_theta': True,    # assoc-angle knob feeds alpha, not Bonnet
        'domain': ('disk', 0.0, 0.999),
        'p_from': lambda order, radius: {
            'n': int(min(max(order + 1, 2), 8)),
            'eps': 0.085 / min(max(radius / 1.2, 0.5), 2.0),
            'alpha': 0.0},
        'count': "Wings (n pairs)",
        'associate': True,           # -> ANGLE_PARAM; the angle knob is alpha
        'mask_punctures': lambda p: [
            (rho, p['eps'])
            for rho in _atower_ends(p['n'], p.get('alpha', 0.0))],
        'clip_punctures': True,       # snap wing rims onto the mask circle
        'clip': False,
        'radial_grade': 'rim',       # fine cells at the |z|=1 ends -> smooth
        'res_boost': (1.7, 1.7),     # unequal wing rims
        'cycles': lambda p: [
            (rho, 0.12)
            for rho in _atower_ends(p['n'], p.get('alpha', 0.0))],
        'cycle_free': (2,),          # vertical translation is the period
        'test_order': 1,             # n = 2, the classical 4-wing saddle
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
        # g = z^k makes the disk-edge flower gain ~2(k+1) lobes, so the
        # angular sampling must grow with the order or the rim reads
        # polygonal at high k; scale nv with k (capped), nu mildly.
        'res_boost': lambda order: (
            1.7 * min(1.0 + 0.12 * (min(max(order + 1, 2), 12) - 2), 1.6),
            1.5 * min(1.0 + 0.45 * (min(max(order + 1, 2), 12) - 2), 2.8)),
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
        'res_boost': (1.4, 1.9),     # rebalanced for the 64 res baseline
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
        'res_boost': (2.0, 2.0),     # annulus double-cover; tuned at 64 res
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
        'res_boost': (1.5, 1.5),     # finer planar-end puncture rims
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
        # Enneper flares like r^{2k+1} toward the disk edge, so a linear grid
        # spends its nodes on the flat centre and leaves the outer rim a
        # coarse polygon (the classic faceted Enneper at default res).
        # Cluster nodes toward the rim and give the ring enough angular
        # samples to read as a smooth curve.  The rim gains ~2(k+1) lobes
        # with order, so the angular multiplier (nv) scales with k -- order 5
        # gets ~3x the angular resolution of order 1 -- keeping the rim
        # smooth at every order, not just order 1 (nu scales mildly, both
        # capped so high k stays sane).
        'radial_grade': 'rim',
        'res_boost': lambda order: (
            1.5 * min(1.0 + 0.12 * (min(max(order, 1), 12) - 1), 1.7),
            1.7 * min(1.0 + 0.5 * (min(max(order, 1), 12) - 1), 3.0)),
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
        # dense rims, smooth flaring wings; rebalanced for the 64 res
        # baseline (was 3.2 x 4.2 at 48) to hold the vert count near 30k
        'res_boost': (2.4, 3.2),
        'cycles': lambda p: [(r, 0.12) for r in p['_rho']],
        'test_order': 1,                         # order 1 -> n = 3 (trinoid)
    },
    # --- M3 batch: genus-0 k-noid symmetry variants + Lopez (harvested) ---
    # Weierstrass data harvested from Matthias Weber's minimalsurfaces.blog
    # (research/msblog_harvest/spheres.json).  These are Lopez-Ros
    # deformations of the Jorge-Meeks k-noid into the symmetry groups of the
    # pyramid, bipyramid, prism and antiprism, together with F. Lopez's
    # genus-0 spheres and a finite piece of a Riemann-type surface.  g and dh
    # are elementary rational maps of z on a disk/annulus domain; the
    # Lopez-Ros scale rho and branch constant a are the closed forms that
    # close every real period (verified < 1e-6 by the period gate below), so
    # each row meshes straight from the numeric radial Weierstrass integrator
    # with the same object-space end clip the Jorge-Meeks KNOID uses.  The
    # helper functions (_m3_*) sit just after this dict.
    # References:
    #   L. P. Jorge & W. H. Meeks III, "The topology of complete minimal
    #     surfaces of finite total Gaussian curvature", Topology 22 (1983)
    #     203-221 (the symmetric k-noids);
    #   F. J. Lopez & A. Ros, "On embedded complete minimal surfaces of
    #     genus zero", J. Differential Geom. 33 (1991) 293-300 (the Lopez-Ros
    #     deformation / vertical-flux parameter);
    #   F. J. Lopez, "The classification of complete minimal surfaces with
    #     total curvature greater than -12 pi", Trans. AMS 334 (1992) 49-74
    #     (the two-ended index-2 spheres);
    #   B. Riemann, "Ueber die Flaeche vom kleinsten Inhalt bei gegebener
    #     Begrenzung", Abh. Koenigl. Ges. Wiss. Goettingen 13 (1867)
    #     (the Riemann minimal example this finite piece is cut from);
    #   H. Karcher, "Construction of minimal surfaces", Surveys in Geometry
    #     (1989), for the symmetrization construction of all four k-noids.
    'M3_PYR': {
        # Pyramidal k-noid: n slanted catenoid ends around a pyramid + one
        # axial end.  g = rho z^{n-1}/(z^n - a^n), dh = z^{n-1}(z^n - a^n)/
        # (z^n - 1)^2 dz; the n ends sit at the n-th roots of unity.  rho is
        # the explicit Lopez-Ros scale _m3_pyr_rho(a, n) that closes all
        # periods for any apex tilt a > 1 (tetrahedroid = most symmetric a).
        'label': "Pyramidal k-noid",
        'family': 'SPHERES',
        'g': lambda z, p: p['rho'] * z ** (p['n'] - 1)
        / (z ** p['n'] - p['a'] ** p['n']),
        'dh': lambda z, p: z ** (p['n'] - 1) * (z ** p['n'] - p['a'] ** p['n'])
        / (z ** p['n'] - 1.0) ** 2,
        'domain': ('disk', 0.0, lambda p: p['r1']),
        'p_from': lambda order, radius: (lambda n: {
            'n': n, 'a': _m3_pyr_a(n), 'rho': _m3_pyr_rho(_m3_pyr_a(n), n),
            'r1': 1.3 + 0.35 * min(max(radius / 1.2, 0.0), 1.4)})(
                int(max(3, min(order + 2, 8)))),
        'radial_grade': 'rim', 'clip': True,
        'res_boost': (1.9, 1.9),
        'count': "Pyramid order (n)",
        'cycles': lambda p: [(np.exp(2j * math.pi * j / p['n']), 0.12)
                             for j in range(p['n'])],
        'test_order': 1,                         # order 1 -> n = 3
    },
    'M3_PRISM': {
        # Prismatic k-noid: 2*nn ends in prism symmetry, one ring at |z| = b
        # (inside the unit circle) and one aligned ring at |z| = 1/b.
        # g = a^nn z^{nn-1}(z^nn - a^-nn)/(z^nn - a^nn), dh as below; the
        # branch constant a = _m3_prism_a(nn, b) is the closed form (Weber's
        # notebook) that closes the period problem, with rho = a^nn.  The
        # cube k-noid is nn = 4, b = sqrt(2 - sqrt 3).
        'label': "Prismatic k-noid",
        'family': 'SPHERES',
        'g': lambda z, p: p['a'] ** p['nn'] * z ** (p['nn'] - 1)
        * (z ** p['nn'] - 1.0 / p['a'] ** p['nn'])
        / (z ** p['nn'] - p['a'] ** p['nn']),
        'dh': lambda z, p: z ** (p['nn'] - 1)
        * (z ** p['nn'] - p['a'] ** p['nn'])
        * (z ** p['nn'] - 1.0 / p['a'] ** p['nn'])
        / ((z ** p['nn'] - p['b'] ** p['nn']) ** 2
           * (z ** p['nn'] - 1.0 / p['b'] ** p['nn']) ** 2),
        'domain': ('disk', 0.0, lambda p: p['r1']),
        'p_from': lambda order, radius: (lambda nn: (lambda b: {
            'nn': nn, 'b': b, 'a': _m3_prism_a(nn, b),
            'r1': 1.0 / b + 0.3})(_m3_prism_b(nn)))(
                int(max(3, min(order + 2, 7)))),
        'radial_grade': 'rim', 'clip': True,
        'res_boost': (1.9, 1.9),
        'count': "Prism order (nn)",
        'cycles': lambda p: _m3_ring_cyc(p['nn'], p['b'], p['a'], 0.0),
        'test_order': 2,                         # order 2 -> nn = 4 (cube)
    },
    'M3_BIPYR': {
        # Bipyramidal k-noid: m equatorial catenoid ends at the m-th roots of
        # unity plus the two axial ends (z = 0 and z = infinity), the apices
        # of a bipyramid over a regular m-gon.  g = (z^m - s^m)/(z(s^m z^m -
        # 1)), dh = (z^m - s^m)(s^m z^m - 1)/((z^m - 1)^2 z) dz; all closed
        # form in the single growth-ratio s (octahedroid = m = 4).  An
        # annulus domain excludes the two axial ends, whose flares the
        # object-space end clip trims.
        'label': "Bipyramidal k-noid",
        'family': 'SPHERES',
        'g': lambda z, p: (z ** p['m'] - p['s'] ** p['m'])
        / (z * (p['s'] ** p['m'] * z ** p['m'] - 1.0)),
        'dh': lambda z, p: (z ** p['m'] - p['s'] ** p['m'])
        * (p['s'] ** p['m'] * z ** p['m'] - 1.0)
        / ((z ** p['m'] - 1.0) ** 2 * z),
        'domain': ('disk', lambda p: p['r0'], lambda p: p['r1']),
        'p_from': lambda order, radius: (lambda m: {
            'm': m, 's': _m3_bipyr_s(m), 'r0': 0.34,
            'r1': 2.9})(int(max(3, min(order + 2, 7)))),
        'radial_grade': 'both', 'clip': True,
        'res_boost': (1.9, 1.9),
        'count': "Bipyramid order (m)",
        'cycles': lambda p: [(0.0, 0.12)]
        + [(np.exp(2j * math.pi * j / p['m']), 0.14) for j in range(p['m'])],
        'test_order': 1,                         # order 1 -> m = 3
    },
    'M3_ANTI5': {
        # Antiprismatic k-noid: 2*nn ends in antiprism symmetry -- one ring
        # at |z| = b and a HALF-STEP-rotated ring at |z| = 1/b (phase pi/nn).
        # g = rho z^{nn-1}(z^nn + a^-nn)/(z^nn - a^nn), dh as below.  Unlike
        # the prism, the period problem here is only closed-form for nn = 2,3
        # and needs a numeric NSolve for higher nn; only the harvested
        # nn = 5, b = 0.2 constants (a, rho) are shipped -- a single verified
        # member (the general family is deferred; see the TODO note).
        'label': "Antiprismatic k-noid (nn=5)",
        'family': 'SPHERES',
        'g': lambda z, p: p['rho'] * z ** (p['nn'] - 1)
        * (z ** p['nn'] + 1.0 / p['a'] ** p['nn'])
        / (z ** p['nn'] - p['a'] ** p['nn']),
        'dh': lambda z, p: z ** (p['nn'] - 1)
        * (z ** p['nn'] - p['a'] ** p['nn'])
        * (z ** p['nn'] + 1.0 / p['a'] ** p['nn'])
        / ((z ** p['nn'] - p['b'] ** p['nn']) ** 2
           * (z ** p['nn'] + 1.0 / p['b'] ** p['nn']) ** 2),
        'domain': ('disk', lambda p: 0.5 * p['b'], lambda p: 1.6 / p['b']),
        'p_from': lambda order, radius: {
            'nn': 5, 'b': 0.2, 'a': 0.2748767946679093,
            'rho': 0.0015692436842339352},
        'radial_grade': 'both', 'clip': True,
        'res_boost': (2.0, 2.0),
        'cycles': lambda p: _m3_ring_cyc(p['nn'], p['b'], p['a'],
                                         math.pi / p['nn']),
        'test_order': 1,
    },
    'M3_LOPEZ': {
        # Lopez sphere with two ends of index 2 (Lopez 1992): an annulus
        # 1/e <= |z| <= e between two winding-2 ends at z = 0 and z =
        # infinity.  g = (z^2 + c i z + 1)/(B z), dh = i(z^2 + c i z + 1)/
        # (B z^2) dz with B = 1/sqrt(2 - c^2); c = 0 is the doubly-covered
        # catenoid, and c in (0, sqrt 2) breaks the symmetry (order steps c).
        'label': "Lopez sphere (2 ends of index 2)",
        'family': 'SPHERES',
        'g': lambda z, p: p['B'] * (z * z + 1j * p['c'] * z + 1.0) / z,
        'dh': lambda z, p: 1j * p['B'] * (z * z + 1j * p['c'] * z + 1.0)
        / (z * z),
        'domain': ('disk', lambda p: 1.0 / p['r1'], lambda p: p['r1']),
        'p_from': lambda order, radius: _m3_lopez_p(order, radius),
        'radial_grade': 'both', 'clip': True,
        'res_boost': (1.6, 1.6),
        'count': "Asymmetry (c steps)",
        'cycles': lambda p: [(0.0, 0.22)],
        'test_order': 1,                         # order 1 -> c = 0
    },
    'M3_CATENN': {
        # Sphere with one catenoid end and one Enneper end (Lopez 1992,
        # total curvature -8 pi): g = rho(z - 1/z), dh = (z - 1/z) dz on the
        # annulus 0.16 <= |z| <= 2.6.  The catenoid end sits at z = 0 (dh
        # simple pole -> logarithmic growth), the Enneper end at z = infinity
        # (dh grows -> quadratic flaring); rho is the free Lopez-Ros scale.
        'label': "Sphere: catenoid + Enneper end",
        'family': 'SPHERES',
        'g': lambda z, p: p['rho'] * (z - 1.0 / z),
        'dh': lambda z, p: (z - 1.0 / z),
        'domain': ('disk', lambda p: p['r0'], lambda p: p['r1']),
        'p_from': lambda order, radius: {
            'rho': 0.4, 'r0': 0.16,
            'r1': 2.6 * min(max(radius / 1.2, 0.5), 1.5)},
        'radial_grade': 'both', 'clip': True,
        'res_boost': (1.8, 1.8),
        'cycles': lambda p: [(0.0, 0.10)],
        'test_order': 1,
    },
    'M3_FRIEM': {
        # Finite Riemann (plane-1 catenoid-2): the compact disk piece of a
        # Riemann-type surface with two catenoid ends at z = +-1 and a
        # higher end toward infinity.  g = t(z^2 + 3)/(z^2 - 1), dh =
        # (z^2 + 3)/(z^2 - 1) dz (the notebook's P/Q form -- authoritative
        # over the page's P*Q); t is the Lopez-Ros tilt.  Illustrates the
        # Lopez-Ros theorem (never embedded).
        'label': "Finite Riemann (plane + 2 catenoids)",
        'family': 'SPHERES',
        'g': lambda z, p: p['t'] * (z * z + 3.0) / (z * z - 1.0),
        'dh': lambda z, p: (z * z + 3.0) / (z * z - 1.0),
        'domain': ('disk', 0.0, lambda p: p['r1']),
        'p_from': lambda order, radius: {
            't': 0.2, 'r1': 1.35 + 0.4 * min(max(radius / 1.2, 0.0), 1.4)},
        'radial_grade': 'rim', 'clip': True,
        'res_boost': (1.7, 1.7),
        'cycles': lambda p: [(1.0, 0.15), (-1.0, 0.15)],
        'test_order': 1,
    },
    # --- M3 batch: higher genus (Chen-Gackstatter) + symmetrizations
    # (harvested) -------------------------------------------------------
    # Two surfaces from the minimalsurfaces.blog harvest that build clean
    # at default settings and pass every gate (period closure < 1e-6, 2 m
    # fit, manifold, conformal UV).  See the TODO(zoo) note below for the
    # harvested higher-genus Chen-Gackstatter towers and the plane-with-
    # catenoids that are deliberately deferred rather than shipped rough.
    'TILT_SCHERK': {
        # Tilted Scherk: the doubly periodic Scherk surface with a Lopez-Ros
        # deformation Rho that tilts its four ends.  g = Rho z,
        # dh = 4z/(z^4 - 1); genus 0, four ends at the 4th roots of unity.
        # The Weierstrass data here (phi / Xexact) drives the period-closure
        # gate.  The OPERATOR meshes it connected and gap-free over the whole
        # Cells U x V block via toolkit._tilt_scherk_doubly: the tilt is
        # exactly a horizontal reparametrization of the classical Scherk graph
        # (height unchanged), verified to machine precision against Xexact, so
        # it tiles as ONE continuous surface with the walls leaned rather than
        # as disconnected wing-end copies.
        # Rho (from the radius slider) is the tilt; Rho = 1 is symmetric
        # (and reproduces SCHERK1 exactly).
        'label': "Tilted Scherk (doubly periodic)",
        'family': 'DOUBLY',
        'phi': _tiltscherk_phi,      # kept for the period-closure gate
        'Xexact': _tiltscherk_X,     # exact immersion (period gate + verify)
        'domain': ('disk', 0.0, 0.999),
        'p_from': lambda order, radius: {
            'Rho': 1.0 + 0.5 * min(max(radius / 1.2, 0.0), 1.6),
            'eps': 0.09},
        'mask_punctures': lambda p: [
            (r, p['eps']) for r in _tiltscherk_roots()],
        'clip_punctures': True,      # snap wing rims onto the mask circle
        'clip': False,
        'radial_grade': 'rim',       # fine cells at the |z| = 1 ends
        'res_boost': (1.6, 1.6),
        'cycles': lambda p: [(r, 0.12) for r in _tiltscherk_roots()],
        'cycle_free': (0, 1),        # the two doubly periodic translations
        'test_order': 1,
    },
    'ENNEPER_NCAT': {
        # Enneper with n catenoids (decorated Enneper): genus 0, one Enneper
        # end at infinity + n catenoidal ends at the n-th roots of unity.
        # Closed-form period solution (Lopez-Ros factor 1, neck b =
        # (2 - a^n)^(1/n)); the free shape parameter a is fixed to 1.05, a
        # value that keeps the ends balanced and tear-free.  n is capped at
        # 3: at n >= 4 the single Enneper end flares fast enough that the
        # object-space end clip shears one wing into a thin sail at the
        # default resolution, so the count stops where the mesh stays clean.
        'label': "Enneper with n Catenoids",
        'family': 'SPHERES',
        'phi': _enc_phi,
        'domain': ('disk', 0.0, lambda p: p['r1']),
        'p_from': lambda order, radius: (lambda n: {
            'n': n, 'a': 1.05, 'b': _enc_b(n, 1.05),
            'r1': 1.3 + 0.25 * min(max(radius / 1.2 - 1.0, 0.0), 2.0)})(
                int(max(2, min(order, 3)))),
        'count': "Catenoids (n)",
        # the catenoid ends sit on |z| = 1; puncture + rim-graded sampling
        # give clean neck rims, the Enneper end flares to the disk edge
        'mask_punctures': lambda p: [(r, 0.16) for r in _enc_roots(p['n'])],
        'clip': True,
        'radial_grade': 'rim',
        'res_boost': (1.9, 2.1),
        'cycles': lambda p: [(r, 0.1) for r in _enc_roots(p['n'])],
        'test_order': 2,             # order 2 -> n = 2
    },
    'JEENER': {
        # Jeener's flower: the Weierstrass data f = z^2, g = z, which
        # integrates in closed form to
        #     x = Re(w^3/3 - w^5/5)
        #     y = Re(i (w^3/3 + w^5/5))
        #     z = Re(w^4/2)
        # -- an algebraic minimal surface of degree 5, explored and
        # engraved ("Petale") by Patrice Jeener.  In this catalog's
        # (g, dh) convention f = z^n becomes dh = 2 g f = 2 z^(n+1),
        # which is what the row carries; n = 2 is Jeener's own case and
        # n = 0 degenerates to Enneper, so the order slider walks a
        # one-parameter family of flowers with more and more petals.
        #
        # There is a BRANCH POINT at w = 0: dh vanishes to order n+1
        # while g has a simple zero, so the induced metric goes like
        # |w|^(2n) and the immersion is branched at the flower's
        # centre.  That is a property of the surface, not a defect --
        # it is why the petals meet in a point rather than a disk --
        # and it is why this row is a plain simply connected disk with
        # no period conditions to close.
        'label': "Jeener's Flower",
        'family': 'CLASSICAL',
        'g': lambda z, p: z,
        'dh': lambda z, p: 2.0 * z ** (p['n'] + 1),
        'domain': ('disk', 0.0, lambda p: p['reach']),
        'p_from': lambda order, radius: {
            'n': int(min(max(order + 1, 1), 10)),
            'reach': min(max(radius, 0.4), 3.0)},
        'count': "Flower order (n)",
        'clip': False,
        # the immersion grows like r^(n+2) toward the disk edge, so the
        # samples must bunch there or the petals read as facets
        'radial_grade': 'rim',
        'res_boost': (1.5, 1.8),
        'associate': True,       # simply connected: every period is 0
        'test_order': 1,         # order 1 -> n = 2, Jeener's own case
    },
    # TODO(zoo) -- harvested but deliberately NOT shipped (would be
    # mislabeled / rough; honesty gate):
    #   * Higher-genus Chen-Gackstatter towers -- both the hyperelliptic
    #     g2/g4/g5 (dh = 1, g = rho*sqrt(z)*prod sqrt(z^2 - a_i^2)/...,
    #     branch constants harvested as literals) and the k-fold-symmetric
    #     symm-chen-gackstatter-gn (g = z^{-e}(1 - z^2)^e, e = (k-1)/k,
    #     closed-form Gamma-quotient rho).  The engine's only higher-genus
    #     assembler (we.tile_dihedral / halfplane_patch) is specialized to
    #     the Costa-Hoffman-Meeks cut structure (cuts along [0,1] and
    #     (-inf,-1], catenoid/planar ends at +-1, a percentile radius clip
    #     sized for finite necks).  Driven with the Chen-Gackstatter data it
    #     produces non-manifold, wrong-genus meshes and its radius clip
    #     shears off the single unbounded Enneper end.  A correct build needs
    #     a dedicated cyclic-cover assembler: integrate one angular wedge
    #     (the notebooks use [0, pi/2]) with an Enneper-end-aware trim, then
    #     rotate/reflect and weld -- a separate reparametrization, left for a
    #     follow-up.  (rho = chm_modulus and the Gamma quotients agree, so
    #     the constants are verified; only the assembler is missing.)
    #   * symm-Costa (n > 2) is NOT a new surface: its closed-form Gamma
    #     rho equals chm_modulus(n-1) exactly (0.955978, 0.988070, 0.995117,
    #     0.997535 for n = 2..5), i.e. symmetrized Costa of order n is the
    #     Costa-Hoffman-Meeks surface of genus n-1 -- already shipped as
    #     COSTA_HM.  Not double-listed.
    #   * Plane-with-catenoids (g = Rho/sqrt(z), dh = 1/(sqrt(z-1)sqrt(z+1)))
    #     has a branched Gauss map and height differential; radial disk rays
    #     cross the sqrt branch cuts on the real axis and tear the surface.
    #     It needs an upper-half-plane integrator with a Schwarz reflection
    #     (the notebook's y in (eps, pi - eps) domain), not yet in the engine.
}


# --- M3 batch helpers (genus-0 k-noid symmetry variants) ------------------
# Lopez-Ros closing constants (rho, a) and period-cycle builders for the
# harvested M3 sphere rows at the end of WE_SURFACES above.  Kept as
# module-level functions so the row lambdas resolve them lazily.

def _m3_pyr_a(n):
    """Pyramidal apex tilt a > 1, gentler with higher order n."""
    return max(1.12, 1.7 - 0.18 * (n - 3))


def _m3_pyr_rho(a, n):
    """Explicit Lopez-Ros scale closing the pyramidal k-noid periods."""
    return math.sqrt((a ** n - 1.0) * (1.0 + a ** n * (n - 1) + n)) \
        / math.sqrt(n - 1)


def _m3_prism_b(nn):
    """Shape parameter b in (0, 1) per prism order (cube = nn 4)."""
    return {2: 0.72, 3: 0.60, 4: math.sqrt(2 - math.sqrt(3)),
            5: 0.45, 6: 0.42, 7: 0.78}.get(nn, 0.5)


def _m3_prism_a(nn, b):
    """Closed-form branch constant a(b, nn) (Weber's notebook) that solves
    the prismatic k-noid period problem symbolically; rho = a**nn."""
    t1 = (b ** (4 * nn) * (1 - 3 * nn) + b ** (2 + 2 * nn) * (1 - 3 * nn)
          + b ** 2 * (nn - 1) + b ** (6 * nn) * (nn - 1)) ** (-1.0 / nn)
    inner = (b ** 2 * (nn - 1) ** 2 + b ** (2 + 4 * nn) * (nn - 1) ** 2
             + b ** 4 * nn ** 2 + b ** (4 * nn) * nn ** 2
             + b ** (2 + 2 * nn) * (4 * nn - 2))
    t2 = (-b ** 2 - b ** (4 * nn) - b ** (2 * nn) * (1 + b ** 2) * (2 * nn - 1)
          + (1 - b ** (2 * nn)) * math.sqrt(inner)) ** (1.0 / nn)
    a = b * t1 * t2
    return a.real if isinstance(a, complex) else a


def _m3_bipyr_s(m):
    """Growth ratio s in (0, 1) per bipyramid order m."""
    return {3: 0.7, 4: 0.8, 5: 0.85, 6: 0.9}.get(m, 0.8)


def _m3_lopez_p(order, radius):
    """Lopez-sphere params: order steps the asymmetry c (c = 0 is the
    doubly-covered catenoid), radius scales the annulus reach."""
    c = min(0.35 * (order - 1), 1.35)
    return {'c': c, 'B': 1.0 / math.sqrt(2.0 - c * c),
            'r1': 3.2 * min(max(radius / 1.2, 0.5), 1.6)}


def _m3_ring_cyc(nn, b, a, twist=0.0):
    """Period cycles around the two rings of ends at |z| = b and |z| = 1/b
    (the outer ring phase-twisted by `twist` -- pi/nn for the antiprism, 0
    for the prism).  Each contour radius is shrunk to clear the nearest
    other singular circle so the residue integral stays clean."""
    out = []
    for R, ph in ((b, 0.0), (1.0 / b, twist)):
        other = [x for x in (b, 1.0 / b, a, 1.0 / a) if abs(x - R) > 1e-6]
        rr = 0.3 * min([abs(x - R) for x in other]
                       + [R * math.sin(math.pi / nn)])
        for j in range(nn):
            out.append((R * np.exp(1j * (TAU * j / nn + ph)), rr))
    return out


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

# The pyramidal / bipyramidal / prismatic k-noids and the Lopez spheres
# (catenoid+Enneper, index-2 two-ended) and Finite Riemann are now the M3
# batch above (harvested Weierstrass data, closed-form Lopez-Ros constants,
# all periods verified < 1e-6).
#
# TODO(zoo) -- deferred (data not fully closing / not yet verified against
# Weber's notebooks; deliberately NOT registered rather than mislabeled):
#   * ANTIPRISMATIC k-noid, general nn: only the single harvested member
#     (nn = 5, b = 0.2 -> a, rho, closes to ~1e-19) is shipped as M3_ANTI5.
#     The nn = 2, 3 "closed forms": the harvest records a(b) for nn = 2 but
#     NOT its companion rho, and a numeric (a, rho) solve at that a bottoms
#     out at ~8e-4 real-period residual (does not close) -- so those sub-
#     cases are withheld pending rho's closed form.  General nn needs a
#     2-parameter NSolve of the residue/period conditions per (nn, b).
#   * Bjorling clothoid (needs a complex Fresnel evaluator)
#   * Karcher's alpha saddle tower is now SADDLE_TOWER_A (one fundamental
#     domain -- one vertical period -- with verified period closure across
#     the whole alpha range).  Still deferred: welded *multi-storey*
#     stacking of the UNEQUAL tower.  The disk unit admits only internal
#     reflection/rotation symmetries (all with zero vertical shift), so no
#     screw deck isometry welds one disk unit onto the next once the wings
#     are unequal; a proper multi-storey mesh needs a translation
#     fundamental domain (cut by two horizontal planes), a separate
#     reparametrization.  The symmetric alpha = 0 tower already stacks via
#     SCHERK_TOWER.
#   * genus-1 helicoid, KMR (Tier 3+); the singly periodic
#     Callahan-Hoffman-Meeks is now shipped as CHM_PERIODIC (appended
#     catalog block at the end of this file)
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
             ANGLE_PARAM=None, STOREY_PARAM=None):
    """Wire every catalog row into the toolkit's registries.  Called
    (with the registry dicts) from minimal_surface_toolkit at import
    time; a bare register() call -- e.g. from the extension loader -- is
    a no-op, since this module has no Blender UI of its own."""
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
        if STOREY_PARAM is not None and spec.get('storeys_label'):
            STOREY_PARAM[key] = spec['storeys_label']
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


# ==========================================================================
# Higher-genus Chen-Gackstatter (appended catalog block)
# ==========================================================================
# Genus-g Chen-Gackstatter surfaces (one winding-3 Enneper end, D2d
# symmetry) on the hyperelliptic curve y^2 = z prod(z^2 - r_i^2), meshed
# watertight (chi = 1 - 2g exactly) from ONE 1/8 Coxeter fundamental
# domain orbited under the full D2d group -- the engine, verified data
# and references live in weierstrass.cg_higher_mesh and the block above
# it.  Genus 2, 4 and 5 are the solved period problems (Chen &
# Gackstatter 1982; E. C. Thayer, Experiment. Math. 4, 1995; data after
# M. Weber, minimalsurfaces.blog).  Genus 3 is deliberately absent: its
# period problem is not solved by this two-parameter normalization.

WE_SURFACES['CG_HIGHER'] = {
    'label': "Chen-Gackstatter (higher genus)",
    'family': 'HIGHER',
    'mesher': we.cg_higher_mesh,
    # the count slider is the target genus; 2, 4, 5 are the solved
    # families, other values snap to the nearest solved genus.  radius
    # scales the spherical end-trim (how far the Enneper end flares).
    'p_from': lambda order, radius: {
        'genus': 2 if order <= 2 else (4 if order <= 4 else 5)},
    'count': "Genus (2/4/5)",
    'test_order': 2,
}
SURFACE_FAMILY['CG_HIGHER'] = 'HIGHER'


# ==========================================================================
# Callahan-Hoffman-Meeks singly periodic surface (appended catalog block)
# ==========================================================================
# The singly periodic analog of the Costa surface (k = 1 member): an
# embedded minimal surface invariant under a vertical translation, two
# horizontal planar ends per translational period (infinitely many ends
# in all), quotient genus 2k + 1 = 3 (one period meshes at chi = -6 with
# the two end punctures -- gated below and in weierstrass).  Weierstrass
# data and the two solved period constants (a, rho) are harvested from
# M. Weber's CHM-(1,1) notebook (research/msblog_harvest/
# singly_periodic.json); the engine, verified branch/strip chart and the
# 16-isometry period assembly live in weierstrass.chm_periodic_mesh and
# the chm_* block above it.  The "Periods" count stacks whole
# translational periods, welded seam-exactly; the radius slider sizes
# the trimmed planar-end disks (how far each flat layer reaches).
#
# References:
#   M. J. Callahan, D. Hoffman, W. H. Meeks III, "Embedded minimal
#     surfaces with an infinite number of ends", Invent. Math. 96 (1989)
#     459-505;
#   M. Weber, https://minimalsurfaces.blog/ (Callahan-Hoffman-Meeks
#     surfaces; the CHM-(1,1) notebook).

WE_SURFACES['CHM_PERIODIC'] = {
    'label': "Callahan-Hoffman-Meeks (singly periodic)",
    'family': 'SINGLY',
    'mesher': we.chm_periodic_mesh,
    # umin is the conformal end-trim: the planar end lives at the strip
    # end u -> -inf and its flare radius grows like ~0.87 e^{-u/4}, so
    # the radius slider slides the cut (bigger radius = wider flat
    # layers relative to the core).
    'p_from': lambda order, radius: {
        'umin': float(np.clip(-5.0 - 2.5 * math.log(
            max(radius, 0.2) / 1.2), -7.5, -3.0))},
    'storeys_label': "Periods",
    'test_order': 1,
}
SURFACE_FAMILY['CHM_PERIODIC'] = 'SINGLY'


# ==========================================================================
# Translation-invariant genus-one helicoid (appended catalog block)
# ==========================================================================
# The helicoid with a handle: the singly periodic minimal surface
# asymptotic to a helicoid whose translational quotient is a rhombic
# torus with two helicoidal ends -- one handle per vertical period.
# Built from the Jacobi theta-function Weierstrass data on C/<1, tau>
# with the fully solved period constants harvested from M. Weber's
# notebook (research/msblog_harvest/singly_periodic.json); the engine,
# verification gates and references (Hoffman-Karcher-Wei 1993/1999;
# Hoffman-Weber-Wolf 2009) live in weierstrass.genus1helicoid_mesh and
# the block above it.  The count slider stacks translational periods
# (the mesh genus equals that count -- verified by the Euler
# characteristic in the self-tests); the radius slider sets how far
# the two helicoidal ends spiral out.

WE_SURFACES['GENUS1_HELICOID'] = {
    'label': "Helicoid with Handle (genus 1)",
    'family': 'SINGLY',
    'mesher': we.genus1helicoid_mesh,
    'p_from': lambda order, radius: {
        'storeys': int(min(max(order, 1), 4))},
    'count': "Periods (handles)",
    'test_order': 1,
}
SURFACE_FAMILY['GENUS1_HELICOID'] = 'SINGLY'


# ==========================================================================
# Symmetrized Chen-Gackstatter towers (appended catalog block)
# ==========================================================================
# The k-fold-symmetric Chen-Gackstatter continuations: one Enneper-type
# end, dihedral-antiprismatic symmetry D_kd (order 4k), on the k-fold
# cyclic cover of the sphere branched over {0, +-r_i, inf}.  Three towers
# (unified data g = rho z^a0 prod (1-(z/r_i)^2)^(+-e), dh = dz with
# e = (k-1)/k), meshed watertight from ONE 1/(4k) quarter-plane
# fundamental piece orbited under the full D_kd group -- the engine,
# closed-form / harvested period constants and references live in
# weierstrass.symmcg_mesh and the block above it:
#   * SYMM_CG      genus k-1   -- the canonical tower; rho is the
#     closed-form Gamma quotient.  k = 2 IS the classical Chen-
#     Gackstatter torus (a cross-check against CHEN_GACK), and k = 4
#     reaches GENUS 3, absent from the CG_HIGHER D2d normalization.
#     Distinct from CG_HIGHER at every shared genus: CG_HIGHER keeps
#     order-8 D2d symmetry with a winding-3 end at every genus, while
#     this tower's symmetry order (4k) and end winding (2k-1) grow
#     with the genus.
#   * SYMM_CG_G2N  genus 2(k-1) -- second tower; branch value a and rho
#     solved numerically (Weber), k in {3, 4, 7, 12}.
#   * SYMM_CG_G3K  genus 3(k-1) -- third tower; branch values a, b from
#     a 2-D period solve (Weber), k in {2, 3, 4, 5, 7}; note k = 2 is a
#     second, different genus-3 surface.
# (Chen & Gackstatter 1982; Karcher's symmetrization method; data after
# M. Weber, minimalsurfaces.blog, "Symmetrized Chen-Gackstatter".)

def _symmcg_snap_k(order, table):
    """Nearest solved symmetry order in `table`."""
    return min(table, key=lambda kk: abs(kk - order))


WE_SURFACES['SYMM_CG'] = {
    'label': "Symmetrized Chen-Gackstatter (k-fold, genus k-1)",
    'family': 'HIGHER',
    'mesher': we.symmcg_mesh,
    # count slider = the symmetry order k (genus k-1); radius scales the
    # spherical end-trim (how far the single Enneper end flares)
    'p_from': lambda order, radius: {
        'tower': 'gn', 'k': int(min(8, max(2, order)))},
    'count': "Symmetry k (genus k-1)",
    'test_order': 4,
}
SURFACE_FAMILY['SYMM_CG'] = 'HIGHER'

WE_SURFACES['SYMM_CG_G2N'] = {
    'label': "Symmetrized Chen-Gackstatter (2-level, genus 2(k-1))",
    'family': 'HIGHER',
    'mesher': we.symmcg_mesh,
    'p_from': lambda order, radius: {
        'tower': 'g2n', 'k': _symmcg_snap_k(order, (3, 4, 7, 12))},
    'count': "Symmetry k (3/4/7/12)",
    'test_order': 3,
}
SURFACE_FAMILY['SYMM_CG_G2N'] = 'HIGHER'

WE_SURFACES['SYMM_CG_G3K'] = {
    'label': "Symmetrized Chen-Gackstatter (3-level, genus 3(k-1))",
    'family': 'HIGHER',
    'mesher': we.symmcg_mesh,
    'p_from': lambda order, radius: {
        'tower': 'g3k', 'k': _symmcg_snap_k(order, (2, 3, 4, 5, 7))},
    'count': "Symmetry k (2/3/4/5/7)",
    'test_order': 2,
}
SURFACE_FAMILY['SYMM_CG_G3K'] = 'HIGHER'




# ==========================================================================
# Catenoid-Enneper and Costa-Wohlgemuth / Wohlgemuth (appended block)
# ==========================================================================
# Tier-3 higher-genus finite-total-curvature surfaces, meshed watertight
# from one fundamental piece orbited under the full point group -- the
# engine, verified period constants and references live in
# weierstrass.cwce_ce_mesh / cwce_cw_mesh and the cwce_* block above
# them:
#   * CATENOID_ENNEPER  genus 2/3/4, ONE catenoid + ONE Enneper end,
#     order-4 symmetry (two orthogonal vertical mirrors); the genus-g
#     member meshes at exactly chi = 2 - 2g - 2 (two open end rims).
#   * COSTA_WOHLGEMUTH  genus 2(k-1), FOUR ends (2 catenoidal +
#     2 planar), prismatic D_kh symmetry; k = 2 is Wohlgemuth's
#     genus-2 surface -- the first complete embedded minimal surface
#     with four ends; chi = 2 - 2 genus - 4.
#   * WOHLGEMUTH_G3     Wohlgemuth's second surface: genus 3, four
#     ends (k = 2 member of the genus-3(k-1) tower; the k > 2 period
#     problems did not close from the blog's constants -- BACKLOG.md).
# (C. J. Costa 1984; M. Wohlgemuth, Bonn dissertation 1993 and Arch.
# Rational Mech. Anal. 137, 1997; Chen & Gackstatter 1982; data after
# M. Weber, minimalsurfaces.blog, higher-symmetry Wohlgemuth notebook
# by Ramazan Yol.)

WE_SURFACES['CATENOID_ENNEPER'] = {
    'label': "Catenoid-Enneper (higher genus)",
    'family': 'HIGHER',
    'mesher': we.cwce_ce_mesh,
    # count slider = the genus (2, 3 or 4 -- the solved period
    # problems); radius scales the two end trims (catenoid funnel
    # depth / Enneper flare reach)
    'p_from': lambda order, radius: {
        'genus': 2 if order <= 2 else (3 if order == 3 else 4)},
    'count': "Genus (2/3/4)",
    'test_order': 2,
}
SURFACE_FAMILY['CATENOID_ENNEPER'] = 'HIGHER'

WE_SURFACES['COSTA_WOHLGEMUTH'] = {
    'label': "Costa-Wohlgemuth (4 ends)",
    'family': 'HIGHER',
    'mesher': we.cwce_cw_mesh,
    # count slider = the symmetry order k (genus 2(k-1)); k snaps to
    # the solved orders 2/3/4/5/7.  radius slides the strip trims
    # (planar-end reach and catenoid depth).
    'p_from': lambda order, radius: {
        'fam': 'cw',
        'k': min((2, 3, 4, 5, 7), key=lambda kk: abs(kk - order))},
    'count': "Symmetry k (2/3/4/5/7)",
    'test_order': 2,
}
SURFACE_FAMILY['COSTA_WOHLGEMUTH'] = 'HIGHER'

WE_SURFACES['WOHLGEMUTH_G3'] = {
    'label': "Wohlgemuth Second Surface (genus 3)",
    'family': 'HIGHER',
    'mesher': we.cwce_cw_mesh,
    'p_from': lambda order, radius: {'fam': 'w2', 'k': 2},
    'test_order': 1,
}
SURFACE_FAMILY['WOHLGEMUTH_G3'] = 'HIGHER'

# ==========================================================================
# Doubly periodic KMR + Wei surfaces (appended catalog block)
# ==========================================================================
# Three doubly periodic four-ended surfaces on the reusable hyperelliptic
# tiler in weierstrass (dperiodic_*): ONE conformal patch is integrated in
# exponential coordinates, its boundary arcs are snapped exactly onto
# their vertical mirror planes / straight lines, and the surface is the
# orbit of that patch under the reflections/rotations those arcs
# generate, tiled at the TRUE lattice vectors and welded seam-exactly.
# The Cells U / V counts of the periodic operator drive the 2-D lattice
# tiling.  Wall-constant closure (the period problem), quotient topology
# (chi = 2 - 2g - 4, four Scherk-type end rims), manifoldness and
# orientability are all measured in the weierstrass self-tests.
#
#   * KMR_DOUBLY   -- the Karcher-Meeks-Rosenberg toroidal Scherk family
#     (genus-1 quotient, 4 parallel Scherk ends).  The count slider
#     walks the branch modulus a (every member closes -- the family's
#     period problem is solved by its reciprocal branch symmetry).
#   * KMR3_DOUBLY  -- the KMR-3 member with a Mobius Gauss map
#     (z+eps)/(z-eps) and elliptic dh: the ends TILT out of the lattice
#     plane (lattice vector T2 = (0, 2c2, 2c3)); assembled from a mirror
#     plane and two straight lines in the surface.
#   * WEI_DOUBLY   -- Fusheng Wei's genus-2 surface: a handle added to
#     the KMR/Scherk picture.  The count slider walks the harvested
#     one-parameter family (b, a) with a solved from b by the period
#     condition Im int_a^b (G + 1/G) dz/z = 0 (verified in the tests).
#
# References (full citations in the weierstrass engine block):
#   Karcher 1988; Meeks-Rosenberg 1989; Perez-Rodriguez-Traizet 2005
#   (the "KMR" classification); F. Wei 1992; data after M. Weber,
#   https://minimalsurfaces.blog/ (KMR-2, KMR-3, Doubly Wei notebooks).
# The KMR-1 notebook publishes only a precomputed mesh (no Weierstrass
# data), so that member is not reconstructable here -- see BACKLOG.md.

def _dp_rmin(radius):
    """End depth from the radius slider: bigger radius digs the four
    Scherk ends deeper (the truncation |z| = rmin moves toward 0)."""
    return float(np.clip(0.05 * (1.2 / max(radius, 0.2)) ** 1.5,
                         0.008, 0.2))


WE_SURFACES['KMR_DOUBLY'] = {
    'label': "Karcher-Meeks-Rosenberg (doubly periodic)",
    'family': 'DOUBLY',
    'mesher': we.dperiodic_mesh,
    'cells2d_mesher': we.dperiodic_mesh,
    'dp_key': 'kmr2',
    'p_from': lambda order, radius: {
        'a': (0.25, 0.32, 0.40, 0.48, 0.55)[
            int(np.clip(order, 1, 5)) - 1],
        'rmin': _dp_rmin(radius)},
    'count': "Family member (a)",
    'test_order': 3,
}
SURFACE_FAMILY['KMR_DOUBLY'] = 'DOUBLY'

WE_SURFACES['KMR3_DOUBLY'] = {
    'label': "Karcher-Meeks-Rosenberg (KMR-3, tilted ends)",
    'family': 'DOUBLY',
    'mesher': we.dperiodic_mesh,
    'cells2d_mesher': we.dperiodic_mesh,
    'dp_key': 'kmr3',
    'p_from': lambda order, radius: {
        'xmin': _dp_rmin(radius)},
    'test_order': 1,
}
SURFACE_FAMILY['KMR3_DOUBLY'] = 'DOUBLY'

WE_SURFACES['WEI_DOUBLY'] = {
    'label': "Wei Doubly Periodic (genus 2)",
    'family': 'DOUBLY',
    'mesher': we.dperiodic_mesh,
    'cells2d_mesher': we.dperiodic_mesh,
    'dp_key': 'wei',
    'p_from': lambda order, radius: (lambda bb, aa: {
        'b': bb, 'a': aa, 'rmin': _dp_rmin(radius)})(
            *we.DPERIODIC_WEI_SAMPLES[int(np.clip(order, 1, 7)) - 1]),
    'count': "Family member (b = .3 ... .7)",
    'test_order': 1,
}
SURFACE_FAMILY['WEI_DOUBLY'] = 'DOUBLY'

# ==========================================================================
# Doubly periodic long tail (appended catalog block)
# ==========================================================================
# Four catalog rows over the dptail_* engine in weierstrass -- the
# remaining doubly periodic repository surfaces with real hyperelliptic
# branch points and dh = dz/z, each shipped with its notebook's solved
# period constants and gated by measured wall residuals + quotient
# topology (chi = 2 - 2 genus - 4, four Scherk end rims) in the
# weierstrass self-tests:
#
#   * KARCHER_SCHERK_DP -- Karcher's doubly periodic Scherk surfaces
#     with handles (genus 2, 3, and the 'exotic' genus 3 branch
#     arrangement): assembled from a straight diagonal line in the
#     surface plus two vertical mirrors on a square 2 pi lattice.
#   * WEI_TOWER_DP -- Fusheng Wei's higher-genus towers (1,3)/(1,4)
#     (with family members walking the solved parameter tables),
#     the less-symmetric (2,3) genus 4, and the (1,6) genus 6 member.
#   * RTW_DP -- Rossman-Thayer-Wohlgemuth M1+ (genus 2, with the
#     period parameter b re-solved from the notebook's own condition)
#     and two members of the M1+- genus 3 family.
#   * CONNOR_DP -- Peter Connor's experimental genus 2/3 surfaces
#     (asymmetric g2 -- Newton-refined here to close its period
#     problem -- and the page 78/80/82/84/85 examples).
#
# References (full citations in the weierstrass engine block):
#   Karcher 1988; Wei 1992; Rossman-Thayer-Wohlgemuth 2000; Connor
#   2018-19 / Connor-Weber 2012; data after M. Weber,
#   https://minimalsurfaces.blog/ (doubly periodic repository).
# Deferred (see BACKLOG.md): Karcher-Scherk g1 (notebook unpublished,
# no Weierstrass data), g4 (dh = dz/(z^2 - r^2), ends mid-edge),
# doubly periodic catenoids (branched z^r Gauss map, hexagonal
# assembly), Lubeck-Batista and CHM g3 (theta-function Gauss maps).

def _dpt_p(keys):
    """p_from factory: order picks the member key; radius drives the
    end truncation (rmin for half-annulus styles, trim factor for the
    full-annulus Connor members)."""
    def p_from(order, radius):
        key = keys[int(np.clip(order, 1, len(keys))) - 1]
        return {'dpt_key': key,
                'rmin': _dp_rmin(radius),
                'trim': float(np.clip(6.0 * (radius / 1.2) ** 2,
                                      1.6, 40.0))}
    return p_from


WE_SURFACES['KARCHER_SCHERK_DP'] = {
    'label': "Karcher-Scherk with Handles (doubly periodic)",
    'family': 'DOUBLY',
    'mesher': we.dptail_mesh,
    'cells2d_mesher': we.dptail_mesh,
    'p_from': _dpt_p(('ksg2', 'ksg3', 'ksg3x')),
    'count': "Member (g2 | g3 | g3 exotic)",
    'test_order': 1,
}
SURFACE_FAMILY['KARCHER_SCHERK_DP'] = 'DOUBLY'

WE_SURFACES['WEI_TOWER_DP'] = {
    'label': "Wei Higher-Genus Tower (doubly periodic)",
    'family': 'DOUBLY',
    'mesher': we.dptail_mesh,
    'cells2d_mesher': we.dptail_mesh,
    'p_from': _dpt_p(('wei13_0', 'wei13_1', 'wei13_2', 'wei13_3',
                      'wei14_0', 'wei14_1', 'wei23', 'wei16')),
    'count': "Member ((1,3) c=.1/.2/.5/.8 | (1,4) d=.5/.7 | "
             "(2,3) | (1,6))",
    'test_order': 3,
}
SURFACE_FAMILY['WEI_TOWER_DP'] = 'DOUBLY'

WE_SURFACES['RTW_DP'] = {
    'label': "Rossman-Thayer-Wohlgemuth (doubly periodic)",
    'family': 'DOUBLY',
    'mesher': we.dptail_mesh,
    'cells2d_mesher': we.dptail_mesh,
    'p_from': _dpt_p(('rtwmp', 'rtwm1pm_0', 'rtwm1pm_1')),
    'count': "Member (M1+ | M1+- d=-.03 | M1+- d=-.1)",
    'test_order': 1,
}
SURFACE_FAMILY['RTW_DP'] = 'DOUBLY'

WE_SURFACES['CONNOR_DP'] = {
    'label': "Connor Experimental (doubly periodic)",
    'family': 'DOUBLY',
    'mesher': we.dptail_mesh,
    'cells2d_mesher': we.dptail_mesh,
    'p_from': _dpt_p(('conn_asym', 'conn78', 'conn80', 'conn82',
                      'conn84', 'conn85')),
    'count': "Member (asym g2 | p78 | p80 | p82 | p84 | p85)",
    'test_order': 1,
}
SURFACE_FAMILY['CONNOR_DP'] = 'DOUBLY'


# ==========================================================================
# Singly periodic long tail (appended catalog block; sptail_* engine)
# ==========================================================================
# Karcher-style singly periodic surfaces from the minimalsurfaces.blog
# harvest (research/msblog_harvest/singly_periodic.json), all built by
# the shared sptail machinery in weierstrass: one fundamental patch
# integrated by compound Gauss-Legendre cells, boundary snapped exactly
# onto its symmetry lines/planes, orbited under the isometry group with
# `storeys` translation or screw copies, welded seam-exactly.  Every
# member's period problem closes with a machine-checked residual and its
# Euler characteristic per period is MEASURED in the self-tests (engine
# gates in weierstrass._selftest(), pipeline gates below):
#   * SP_SIX_SCHERK    genus 0, 6 ends/period (2 horizontal + 4 at the
#     end angle phi); rho, a closed forms in phi; translation from the
#     om2 residue at z = 0.  chi/period = -4.  (The obtuse phi > 90 deg
#     branch has a different symmetry group -- deferred, BACKLOG.md.)
#   * SP_ALT_FENCE     alternating fence of half-catenoids, genus 1,
#     2 ends/period, rho = 1/sqrt(a); chi/period = -2.
#   * SP_FENCE_CAT     Karcher's fence of catenoids (translation-
#     invariant catenoid), genus 0, 2 ends/period; rho = sqrt(a).
#   * SP_HELICOIDAL_SCHERK  helicoidal Karcher-Scherk: the saddle tower
#     deformed by a screw motion (twist knob = the associate-angle
#     slider); one modulus R solved on the geometric closure residual;
#     rise = pi R^2/(1+R^4) reproduced to 1e-14; chi/period = 2 - 2k.
#   * SP_ENNEPER_3ANN  translation-invariant Enneper with three annular
#     ends (limit member, notebook by Ramazan Yol): closed-form
#     immersion, no constants; MEASURED genus 0 with 4 ends/period
#     (chi = -2; the harvest's genus-1 annotation contradicts its own
#     rational g, dh -- see the weierstrass block note).
#   * SP_PERIODIC_ENNEPER  the classical periodic Enneper surface
#     (g = z, dh = dz on the universal cover of the punctured disk);
#     closed form, "Turns" stacks whole periods.
#   * SP_SCHERK_ENNEPER  Scherk-Enneper interpolation family (rational
#     g, dh; 2k wing ends on the unit circle + 2 Enneper-type ends);
#     horizontal periods close to machine precision (cycles gate).
# Karcher's symmetrized Scherk towers (the harvest's symmetrized_scherk
# row) are already shipped as SCHERK_TOWER / SADDLE_TOWER_A.
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988);
#   H. F. Scherk (1835); A. Enneper (1864);
#   M. Weber, https://minimalsurfaces.blog/ (6-Ended Scherk g0;
#     Alternating Fence of Half-Catenoids, 2024; Fence of Catenoids;
#     Helicoidal Karcher-Scherk; Periodic Enneper; Enneper-Scherk;
#     Translation-Invariant Torus with 1 Enneper and 3 Annular Ends,
#     notebook by Ramazan Yol, 2024).

WE_SURFACES['SP_SIX_SCHERK'] = {
    'label': "Six-Ended Scherk Tower",
    'family': 'SINGLY',
    'mesher': we.sptail_six_mesh,
    # count slider walks the end angle phi = 10 + 10 order degrees
    # (20..80); radius slides the wing end trims.
    'p_from': lambda order, radius: {
        'phid': 10.0 + 10.0 * int(np.clip(order, 1, 7)),
        'rmax': float(np.clip(30.0 * (radius / 1.2) ** 2, 8.0, 120.0))},
    'count': "End angle (x10 deg + 10)",
    'storeys_label': "Periods",
    'test_order': 2,
}
SURFACE_FAMILY['SP_SIX_SCHERK'] = 'SINGLY'

WE_SURFACES['SP_ALT_FENCE'] = {
    'label': "Alternating Fence of Half-Catenoids",
    'family': 'SINGLY',
    'mesher': we.sptail_fencealt_mesh,
    # count slider walks the neck modulus a; radius digs the two
    # catenoid ends deeper (smaller rmin trim).
    'p_from': lambda order, radius: {
        'a': (1.4, 1.7, 2.0, 2.5, 3.0, 4.0)[
            int(np.clip(order, 1, 6)) - 1],
        'rmin': float(np.clip(0.06 * (1.2 / max(radius, 0.2)) ** 1.2,
                              0.015, 0.2))},
    'count': "Neck modulus (a)",
    'storeys_label': "Periods",
    'test_order': 3,
}
SURFACE_FAMILY['SP_ALT_FENCE'] = 'SINGLY'

WE_SURFACES['SP_FENCE_CAT'] = {
    'label': "Fence of Catenoids (Karcher)",
    'family': 'SINGLY',
    'mesher': we.sptail_fencecat_mesh,
    # count slider walks the neck modulus a (0 < a < 1); radius digs
    # the catenoid funnels deeper (bigger r1 trim ratio).
    'p_from': lambda order, radius: {
        'a': (0.08, 0.14, 0.2, 0.3, 0.42, 0.55)[
            int(np.clip(order, 1, 6)) - 1],
        # keep the funnel flare comparable to the neck spacing, or the
        # catenoid ends engulf the chain and the fence reads as a disk
        'r1': float(np.clip(1.7 * (radius / 1.2) ** 1.5, 1.25, 8.0))},
    'count': "Neck modulus (a)",
    'storeys_label': "Periods",
    'test_order': 3,
}
SURFACE_FAMILY['SP_FENCE_CAT'] = 'SINGLY'

WE_SURFACES['SP_HELICOIDAL_SCHERK'] = {
    'label': "Helicoidal Karcher-Scherk (twisted tower)",
    'family': 'SINGLY',
    'mesher': we.sptail_hks_mesh,
    # count = wings k; the associate-angle knob is the TWIST modulus
    # (0 -> gentle twist, pi/2 -> the strongest solved twist); radius
    # slides the 2k wing end trims.
    'p_from': lambda order, radius: {
        'k': int(np.clip(order + 1, 2, 8)),
        'xmax': float(np.clip(4.0 * (radius / 1.2), 2.0, 8.0))},
    'count': "Wings (k pairs)",
    'associate': True,
    'storeys_label': "Periods",
    'test_order': 3,
}
SURFACE_FAMILY['SP_HELICOIDAL_SCHERK'] = 'SINGLY'

WE_SURFACES['SP_ENNEPER_3ANN'] = {
    'label': "Translation-Invariant Enneper (3 annular ends)",
    'family': 'SINGLY',
    'mesher': we.sptail_e3a_mesh,
    # radius slides the Enneper end trim
    'p_from': lambda order, radius: {
        'Rt': float(np.clip(2.2 * (radius / 1.2) ** 0.7, 1.5, 3.2))},
    'storeys_label': "Periods",
    'test_order': 1,
}
SURFACE_FAMILY['SP_ENNEPER_3ANN'] = 'SINGLY'

WE_SURFACES['SP_PERIODIC_ENNEPER'] = {
    'label': "Periodic Enneper",
    'family': 'SINGLY',
    'mesher': we.sptail_penneper_mesh,
    # radius slides the outer (Enneper) rim; "Turns" stacks whole
    # translational periods of the universal cover
    'p_from': lambda order, radius: {
        'rmax': float(np.clip(1.3 * (radius / 1.2) ** 0.8, 0.9, 2.0))},
    'storeys_label': "Turns (periods)",
    'test_order': 1,
}
SURFACE_FAMILY['SP_PERIODIC_ENNEPER'] = 'SINGLY'


def _sp_se_phi(z, p):
    """Scherk-Enneper (rational Weierstrass data, Weber's notebook):
    G = c z^(k-1)(z^2k - a^2k)/(z^2k - a^-2k),
    dh = i z^(k-1)(z^2k - a^2k)(z^2k - a^-2k)/(z^2k - 1)^3 dz."""
    k, a = p['k'], p['a']
    c = (1.0 - a ** (-2 * k)) / (1.0 - a ** (2 * k))
    zk = z ** (2 * k)
    G = c * z ** (k - 1) * (zk - a ** (2 * k)) / (zk - a ** (-2 * k))
    dh = 1j * z ** (k - 1) * (zk - a ** (2 * k)) \
        * (zk - a ** (-2 * k)) / (zk - 1.0) ** 3
    return (0.5 * (1.0 / G - G) * dh, 0.5j * (1.0 / G + G) * dh, dh)


WE_SURFACES['SP_SCHERK_ENNEPER'] = {
    'label': "Scherk-Enneper",
    'family': 'SINGLY',
    'phi': _sp_se_phi,
    'domain': ('disk', 0.0, lambda p: p['rmax']),
    # count = wing pairs k; a = 0.75 fixed (mid-family member: the
    # a -> 1 limit degenerates toward the plain saddle tower with
    # vanishing wing amplitude).  radius slides the chart reach.
    'p_from': lambda order, radius: {
        'k': int(np.clip(order, 1, 6)), 'a': 0.75,
        'rmax': float(np.clip(2.0 * (radius / 1.2) ** 0.8, 1.4, 3.2))},
    'count': "Wing pairs (k)",
    'clip': False,
    # the wing eps balances the (z - z0)^-2 end divergence against the
    # Enneper core: too small and the 2k wing blades dwarf everything
    'mask_punctures': lambda p: [
        (np.exp(1j * math.pi * j / p['k']),
         min(0.34, 0.7 * math.sin(math.pi / (2 * p['k']))))
        for j in range(2 * p['k'])],
    'res_boost': (1.5, 1.9),
    # horizontal periods vanish around every wing end; the vertical
    # component carries the alternating +-T translation (cycle_free)
    'cycles': lambda p: [
        (np.exp(1j * math.pi * j / p['k']),
         0.4 * min(0.34, 0.7 * math.sin(math.pi / (2 * p['k']))))
        for j in range(2 * p['k'])],
    'cycle_free': (2,),
    'test_order': 2,
}
SURFACE_FAMILY['SP_SCHERK_ENNEPER'] = 'SINGLY'


# ==========================================================================
# SYMM/NONORIENT TAIL (appended catalog block)
# ==========================================================================
# Two groups of rows on the symtail_* engine block in weierstrass:
#
# 1) Symmetrization remainder (genus-0 k-noid variants, family SPHERES)
#    -- the members of minimalsurfaces.blog's "Symmetrizations" index
#    genuinely NOT already in this catalog (see the skip list in the
#    self-tests): the symmetrized finite Riemann (1 planar + 2m
#    catenoid ends), the symmetrized double Enneper (two mutually
#    rotated Enneper ends), k-noids with Enneper ends, and the FULL
#    antiprismatic k-noid family (the catalog had only the harvested
#    nn = 5 member M3_ANTI5; the engine now solves the Lopez-Ros
#    period problem numerically for every nn).
# 2) Non-orientable remainder (family NONORIENT): Henneberg's classical
#    one-sided surface meshed as its actual quotient (a cross-cap weld,
#    not the orientable double-cover patch the CLASSICAL row shows),
#    Kusner's projective planes with p planar ends, and F. J. Lopez's
#    one-ended minimal Klein bottle.  Every non-orientable row is
#    measurably one-sided (orientation propagation meets a
#    contradiction) -- gated in the self-tests below.
#
# References:
#   B. Riemann (1867) and F. J. Lopez, A. Ros, J. Differential Geom. 33
#     (1991) for the (never embedded) finite Riemann family; the
#     symmetrized member follows M. Weber, minimalsurfaces.blog,
#     "Symmetrized Finite Riemann".
#   H. Karcher, "Construction of minimal surfaces" (1989) for the
#     symmetrization method (double Enneper, k-noid families);
#     L. P. Jorge, W. H. Meeks III, Topology 22 (1983) for the k-noids;
#     data after M. Weber, minimalsurfaces.blog ("Symmetrized Double
#     Enneper", "k-Noids with Enneper Ends", "Antiprismatic k-Noids").
#   L. Henneberg (1875); R. Kusner, Bull. Amer. Math. Soc. 17 (1987)
#     291-295; F. J. Lopez, Duke Math. J. 71 (1993) 23-30 -- full
#     citations in the weierstrass symtail engine block.


def _symtail_friem_rho(m, a):
    """Closed-form Lopez-Ros scale of the symmetrized finite Riemann
    surface (Weber's notebook): all periods close for any a in (0,1)."""
    a2m = a ** (2 * m)
    return math.sqrt(1.0 - 2.0 * a2m + a2m * a2m + 2.0 * m
                     - 2.0 * a2m * a2m * m)


def _symtail_dblenn_phi(z, p):
    """Symmetrized double Enneper: g = P/Q, eta = P Q / z^(2n+2) with
    zeta = R0 e^(i pi/(2n+2)) twisting the two Enneper ends against
    each other; all residues at z = 0 vanish identically."""
    n, R0 = p['n'], p['R0']
    zeta1 = (R0 * np.exp(1j * math.pi / (2 * n + 2))) ** (n + 1)
    P = -(zeta1 / (1.0 + zeta1 * zeta1)) * z ** n \
        * (z ** (n + 1) - zeta1)
    Q = z ** (n + 1) - 1.0 / zeta1
    w = z ** (2 * n + 2)
    return (0.5 * (Q * Q - P * P) / w,
            0.5j * (Q * Q + P * P) / w,
            P * Q / w)


def _symtail_kusner_p(order):
    return 2 * int(min(max(order, 1), 3)) + 1          # p = 3, 5, 7


def _symtail_kusner_ends(p_sym):
    """The p inner planar-end punctures of Kusner's projective plane
    (the other p ends are their antipodes outside the unit disk)."""
    s = math.sqrt(2 * p_sym - 1)
    r_in = ((p_sym - s) / (p_sym - 1)) ** (1.0 / p_sym)
    return [r_in * np.exp(2j * math.pi * j / p_sym)
            for j in range(p_sym)]


def _symtail_kusner_eps(p_sym):
    """Per-p puncture radius: inside the rim gap and the end spacing."""
    s = math.sqrt(2 * p_sym - 1)
    r_in = ((p_sym - s) / (p_sym - 1)) ** (1.0 / p_sym)
    return min(0.45 * (1.0 - r_in),
               0.30 * TAU * r_in / p_sym, 0.14)


WE_SURFACES['SYMM_FRIEM'] = {
    # Symmetrized finite Riemann: one planar end (z = infinity) + 2m
    # catenoidal ends at the 2m-th roots of unity, m-fold dihedral
    # symmetry.  g = rho z^(m-1)/(z^2m - a^2m), dh = z^(m-1)
    # (z^2m - a^2m)/(z^2m - 1)^2 dz; rho is the closed form above, the
    # neck parameter a is free.  Like every Riemann-type surface it is
    # never embedded (Lopez-Ros); m = 1 is the M3_FRIEM member already
    # shipped, so the slider starts at m = 2.
    'label': "Symmetrized Finite Riemann (2m catenoids)",
    'family': 'SPHERES',
    'g': lambda z, p: p['rho'] * z ** (p['m'] - 1)
    / (z ** (2 * p['m']) - p['a'] ** (2 * p['m'])),
    'dh': lambda z, p: z ** (p['m'] - 1)
    * (z ** (2 * p['m']) - p['a'] ** (2 * p['m']))
    / (z ** (2 * p['m']) - 1.0) ** 2,
    'domain': ('disk', 0.0, lambda p: p['r1']),
    'p_from': lambda order, radius: (lambda m: {
        'm': m, 'a': 0.9, 'rho': _symtail_friem_rho(m, 0.9),
        'r1': 1.35 + 0.4 * min(max(radius / 1.2, 0.0), 1.4)})(
            int(max(2, min(order, 6)))),
    'count': "Symmetry (m)",
    'radial_grade': 'rim', 'clip': True,
    'res_boost': (1.9, 1.9),
    'cycles': lambda p: [(np.exp(1j * math.pi * j / p['m']), 0.04)
                         for j in range(2 * p['m'])]
    + [(p['a'] * np.exp(1j * math.pi * j / p['m']), 0.03)
       for j in range(2 * p['m'])],
    'test_order': 2,
}
SURFACE_FAMILY['SYMM_FRIEM'] = 'SPHERES'

WE_SURFACES['SYMM_DBLENN'] = {
    # Symmetrized double Enneper: two higher-order Enneper ends (z = 0
    # and z = infinity) rotated against each other by the twist phase of
    # zeta = R0 e^(i pi/(2n+2)).  All periods vanish identically (the
    # residues at 0 cancel by construction, verified in the period
    # gate).  n = 1 is close to the classical DOUBLE_ENNEPER; the
    # slider starts at n = 2 for the genuinely symmetrized members.
    'label': "Symmetrized Double Enneper",
    'family': 'SPHERES',
    'phi': _symtail_dblenn_phi,
    'domain': ('disk', lambda p: 1.0 / p['r1'], lambda p: p['r1']),
    'p_from': lambda order, radius: (lambda n: {
        'n': n, 'R0': 4.0 + 1.0 * (n - 2),
        'r1': 2.0 + 0.5 * min(max(radius / 1.2, 0.0), 2.0)})(
            int(max(2, min(order, 6)))),
    'count': "Symmetry (n)",
    'radial_grade': 'both', 'clip': True,
    'res_boost': (1.6, 2.0),
    'cycles': lambda p: [(0.0, 1.0)],
    'test_order': 2,
}
SURFACE_FAMILY['SYMM_DBLENN'] = 'SPHERES'

WE_SURFACES['KNOID_ENN_ENDS'] = {
    # k-noid with Enneper ends: k ends at the k-th roots of unity whose
    # height differential has POLES OF ORDER FOUR (winding Enneper
    # ends) -- genuinely distinct from both the Jorge-Meeks KNOID
    # (simple catenoid ends) and the shipped ENNK (order-three poles).
    # g = z^(k-1)(z^k - R^k)/(1 - R^k z^k), dh = (1 - (z^k + z^-k)
    # /(R^k + R^-k)) / (z (z^k + z^-k - 2)^2) dz; R is a free squeeze
    # parameter and every period closes identically (period gate).
    'label': "k-Noid with Enneper Ends",
    'family': 'SPHERES',
    'g': lambda z, p: z ** (p['k'] - 1)
    * (z ** p['k'] - p['R'] ** p['k'])
    / (1.0 - p['R'] ** p['k'] * z ** p['k']),
    'dh': lambda z, p: (1.0 - (z ** p['k'] + z ** (-p['k']))
                        / (p['R'] ** p['k'] + p['R'] ** (-p['k'])))
    / (z * (z ** p['k'] + z ** (-p['k']) - 2.0) ** 2),
    # ENNK-style domain: stop just INSIDE the unit circle where the k
    # ends live, and cut each end with a puncture-mask disk -- the
    # order-four Enneper flares then read as clean winding wings
    # instead of the untrimmable blade an outside-reaching domain
    # produces.  The radius slider squeezes the ends via the mask size.
    'domain': ('disk', 0.0, 0.985),
    'p_from': lambda order, radius: {
        'k': int(max(3, min(order, 7))), 'R': 2.0,
        'eps': 0.16 / min(max(radius / 1.2, 0.7), 1.6)},
    'count': "Ends (k)",
    'mask_punctures': lambda p: [
        (np.exp(2j * math.pi * j / p['k']), p['eps'])
        for j in range(p['k'])],
    'radial_grade': 'rim', 'clip': False,
    'res_boost': (2.0, 2.4),
    'cycles': lambda p: [(np.exp(2j * math.pi * j / p['k']), 0.1)
                         for j in range(p['k'])] + [(0.0, 0.3)],
    'test_order': 3,
}
SURFACE_FAMILY['KNOID_ENN_ENDS'] = 'SPHERES'

_SYMTAIL_AP_B = {3: 0.60, 4: 0.55, 5: 0.45, 6: 0.42, 7: 0.40}
# per-order outer reach: how far past the |z| = 1/b end ring the domain
# extends before the object-space clip trims -- tuned so the trimmed
# mesh stays ONE component at every order (gated below)
_SYMTAIL_AP_ROUT = {3: 1.6, 4: 1.4, 5: 1.4, 6: 1.25, 7: 1.25}

WE_SURFACES['ANTIPRISM_KNOID'] = {
    # Antiprismatic k-noid, FULL family: 2*nn catenoid ends in antiprism
    # symmetry (one ring at |z| = b, one half-step-rotated ring at
    # |z| = 1/b).  The branch constant a and Lopez-Ros scale rho have no
    # closed form for general nn; we.symtail_antiprism_constants solves
    # the two-ring period problem numerically per (nn, b) -- at nn = 5,
    # b = 0.2 it reproduces Weber's harvested constants to 1e-9 (the
    # M3_ANTI5 row keeps that exact member).
    'label': "Antiprismatic k-noid (full family)",
    'family': 'SPHERES',
    'g': lambda z, p: p['rho'] * z ** (p['nn'] - 1)
    * (z ** p['nn'] + 1.0 / p['a'] ** p['nn'])
    / (z ** p['nn'] - p['a'] ** p['nn']),
    'dh': lambda z, p: z ** (p['nn'] - 1)
    * (z ** p['nn'] - p['a'] ** p['nn'])
    * (z ** p['nn'] + 1.0 / p['a'] ** p['nn'])
    / ((z ** p['nn'] - p['b'] ** p['nn']) ** 2
       * (z ** p['nn'] + 1.0 / p['b'] ** p['nn']) ** 2),
    'domain': ('disk', lambda p: 0.5 * p['b'],
               lambda p: _SYMTAIL_AP_ROUT[p['nn']] / p['b']),
    'p_from': lambda order, radius: (lambda nn: {
        'nn': nn, 'b': _SYMTAIL_AP_B[nn]})(
            int(max(3, min(order + 2, 7)))),
    'solve': lambda p: dict(p, **dict(zip(('a', 'rho'),
                                          we.symtail_antiprism_constants(
                                              p['nn'], p['b'])))),
    'count': "Antiprism order (nn)",
    'radial_grade': 'both', 'clip': True,
    'res_boost': (2.0, 2.0),
    'cycles': lambda p: _m3_ring_cyc(p['nn'], p['b'], p['a'],
                                     math.pi / p['nn']),
    'test_order': 1,                             # order 1 -> nn = 3
}
SURFACE_FAMILY['ANTIPRISM_KNOID'] = 'SPHERES'


def _symtail_henneberg_X(z, p, theta=0.0):
    """Exact antiderivative of Henneberg's Weierstrass data g = z,
    dh = 2 z (1 - z^-4) dz (halved scale).  Satisfies the antipodal
    identity X(-1/conj z) = X(z) exactly -- the surface is one-sided."""
    F1 = z - z ** 3 / 3.0 + z ** (-3) / 3.0 - 1.0 / z
    F2 = 1j * (z + z ** 3 / 3.0 + z ** (-3) / 3.0 + 1.0 / z)
    F3 = z * z + z ** (-2)
    return np.real(F1), np.real(F2), np.real(F3)


WE_SURFACES['HENNEBERG_RP2'] = {
    # Henneberg's surface meshed as its true one-sided quotient: the
    # complete surface is a once-punctured projective plane (Henneberg
    # 1875 -- the first known non-orientable minimal surface); the
    # annulus 1 <= |z| <= r1 is a fundamental domain of the free
    # antipodal involution z -> -1/conj(z), and welding the |z| = 1 rim
    # to itself antipodally produces a genuine Mobius-strip mesh
    # (chi = 0, one boundary loop, no consistent orientation -- all
    # measured in the self-tests).  The four branch points of the
    # classical immersion (z = +-1, +-i) sit on the welded rim.  The
    # CLASSICAL family's HENNEBERG row shows the familiar orientable
    # double-cover patch; this row is the surface's actual topology.
    'label': "Henneberg (one-sided)",
    'family': 'NONORIENT',
    'g': lambda z, p: z,                  # for the period-closure gate
    'dh': lambda z, p: 2.0 * z * (1.0 - z ** (-4)),
    'Xexact': _symtail_henneberg_X,
    'mesher': we.symtail_crosscap_mesh,
    'crosscap_rim': 'inner',
    'domain': ('disk', 1.0, lambda p: p['r1']),
    'p_from': lambda order, radius: {
        'r1': 1.9 + 0.5 * min(max(radius / 1.2, 0.0), 2.0)},
    'radial_grade': 'rim',
    'clip': False,
    'res_boost': (1.7, 2.2),
    'cycles': lambda p: [(0.0, 1.2)],
    'test_order': 1,
}
SURFACE_FAMILY['HENNEBERG_RP2'] = 'NONORIENT'

WE_SURFACES['KUSNER_RP2'] = {
    # Kusner's projective planes (Kusner 1987): an immersed minimal
    # sphere with 2p planar ends whose immersion commutes with the
    # antipodal map for ODD p, descending to RP^2 with p planar ends
    # (p = 3 is the surface whose inversion is Bryant's Boy surface).
    # G = z^(p-1)(z^p - s)/(s z^p + 1), s = sqrt(2p - 1), dh = i
    # z^(p-1)(z^p - s)(1 + s z^p)/(z^2p + 2 s z^p/(p-1) - 1)^2 dz; all
    # 2p residues vanish (gated), so the immersion is single-valued
    # with NO period problem.  The unit disk is a fundamental domain of
    # z -> -1/conj(z) (p ends inside, their antipodes outside); the
    # cross-cap weld of the rim makes the mesh measurably one-sided
    # with chi = 1 - p and p planar-end boundary loops.
    'label': "Kusner Projective Plane (p planar ends)",
    'family': 'NONORIENT',
    'g': lambda z, p: z ** (p['p'] - 1)
    * (z ** p['p'] - p['s']) / (p['s'] * z ** p['p'] + 1.0),
    'dh': lambda z, p: 1j * z ** (p['p'] - 1)
    * (z ** p['p'] - p['s']) * (1.0 + p['s'] * z ** p['p'])
    / (z ** (2 * p['p']) + 2.0 * p['s'] * z ** p['p']
       / (p['p'] - 1) - 1.0) ** 2,
    'mesher': we.symtail_crosscap_mesh,
    'crosscap_rim': 'outer',
    'domain': ('disk', 0.0, 1.0),
    'p_from': lambda order, radius: (lambda pp: {
        'p': pp, 's': math.sqrt(2 * pp - 1)})(_symtail_kusner_p(order)),
    'count': "Planar ends (3/5/7)",
    'mask_punctures': lambda p: [
        (e, _symtail_kusner_eps(p['p']))
        for e in _symtail_kusner_ends(p['p'])],
    'res_boost': (2.2, 2.4),
    'clip': False,
    'cycles': lambda p: [
        (e, 0.5 * _symtail_kusner_eps(p['p']))
        for e in _symtail_kusner_ends(p['p'])],
    'test_order': 1,                             # order 1 -> p = 3
}
SURFACE_FAMILY['KUSNER_RP2'] = 'NONORIENT'

WE_SURFACES['LOPEZ_KLEIN'] = {
    # F. J. Lopez's one-ended minimal Klein bottle (Duke Math. J. 71,
    # 1993): the unique-in-its-class complete non-orientable minimal
    # surface of total curvature -8 pi with Klein bottle topology.
    # Assembled by the symtail engine from one conformal patch of the
    # orientation double cover and the surface's two straight lines
    # (x- and y-axis 180-degree rotations); the |x| = 1 rim carries the
    # Klein-deck gluing.  chi = -1 with ONE boundary loop (the trimmed
    # end) and no consistent orientation -- all measured.
    'label': "Lopez Minimal Klein Bottle",
    'family': 'NONORIENT',
    'mesher': we.symtail_lopez_klein_mesh,
    'p_from': lambda order, radius: {
        'rmax': 2.6 + 0.6 * min(max(radius / 1.2, 0.4), 2.0)},
    'res_boost': (1.5, 1.5),
    'test_order': 1,
}
SURFACE_FAMILY['LOPEZ_KLEIN'] = 'NONORIENT'


# ==========================================================================
# SP SCHERK FAMILY (appended catalog block)
# ==========================================================================
# Higher-genus singly periodic Scherk towers on the sscherk_* engine
# block in weierstrass: the notebook-solved period constants (extracted
# from the raw minimalsurfaces.blog notebooks, re-verified numerically
# at import of the self-tests) drive four hyperelliptic towers.  Every
# row is meshed from one conformal fundamental patch, snapped onto its
# measured mirror planes, orbited and welded bitwise-exactly; the
# quotient topology (chi = 2 - 2 genus - #ends) is MEASURED in the
# self-tests:
#   * SP_SIX_SCHERK_G1    genus 1, 6 ends/period (chi = -6): the
#     six-ended tower of SP_SIX_SCHERK with a handle; count slider
#     walks the notebook family v4.
#   * SP_COSTA_SCHERK_G1  genus 1, 6 ends/period (chi = -6): the
#     Costa-Scherk tower (handle forming Costa-like saddles); count
#     walks the branch parameter a.
#   * SP_EIGHT_SCHERK_G2  genus 2, 8 ends/period (chi = -10); count
#     walks the end-pair spacing b; the translation reproduces the
#     notebook's closed form transy to ~1e-9.
#   * SP_DASILVA_BATISTA  daSilva-Batista surface (2009), genus 2 with
#     8 ends/period (chi = -10); count walks the FindRoot family.
#
# References:
#   H. Karcher, Manuscripta Math. 62 (1988); H. F. Scherk (1835);
#   K. Li thesis lineage (6/8-ended towers); L. daSilva, V. Ramos
#   Batista (2009); M. Weber, https://minimalsurfaces.blog/ notebooks
#   Singly_6ended_Scherk_g1.nb, Singly_CostaScherk_g1.nb,
#   Singly_8ended_Scherk_g2.nb, Singly_daSilvaBatista_g2.nb
#   (research/msblog_harvest/singly_periodic.json).  Full scholarly
#   details in the weierstrass sscherk block header.

WE_SURFACES['SP_SIX_SCHERK_G1'] = {
    'label': "Six-Ended Scherk Tower (genus 1)",
    'family': 'SINGLY',
    'mesher': we.sscherk_six1_mesh,
    # count walks the notebook members v4 = 0.4 .. 0.95; radius slides
    # the wing-end trim depth
    'p_from': lambda order, radius: {'build_kw': {
        'r2': float(np.clip(8.0 * (radius / 1.2) ** 0.8, 5.0, 10.0))}},
    'count': "Family member (v4 table)",
    'storeys_label': "Periods",
    'test_order': 4,
}
SURFACE_FAMILY['SP_SIX_SCHERK_G1'] = 'SINGLY'

WE_SURFACES['SP_COSTA_SCHERK_G1'] = {
    'label': "Costa-Scherk Tower (genus 1)",
    'family': 'SINGLY',
    'mesher': we.sscherk_costa_mesh,
    'p_from': lambda order, radius: {'build_kw': {
        'r2': float(np.clip(8.0 * (radius / 1.2) ** 0.8, 5.0, 10.0))}},
    'count': "Family member (a table)",
    'storeys_label': "Periods",
    'test_order': 2,
}
SURFACE_FAMILY['SP_COSTA_SCHERK_G1'] = 'SINGLY'

WE_SURFACES['SP_EIGHT_SCHERK_G2'] = {
    'label': "Eight-Ended Scherk Tower (genus 2)",
    'family': 'SINGLY',
    'mesher': we.sscherk_eight_mesh,
    # radius slides both end-trim depths on the log scale
    'p_from': lambda order, radius: (lambda rr: {'build_kw': {
        'rmin': 10.0 ** (-3.0 * rr), 'rmax': 10.0 ** (4.0 * rr)}})(
        float(np.clip((radius / 1.2) ** 0.8, 0.6, 1.4))),
    'count': "Family member (b table)",
    'storeys_label': "Periods",
    'test_order': 3,
}
SURFACE_FAMILY['SP_EIGHT_SCHERK_G2'] = 'SINGLY'

WE_SURFACES['SP_DASILVA_BATISTA'] = {
    'label': "daSilva-Batista Surface (genus 2)",
    'family': 'SINGLY',
    'mesher': we.sscherk_das_mesh,
    'p_from': lambda order, radius: (lambda rr: {'build_kw': {
        'cut1': 10.0 ** (-1.7 * rr), 'cut2': 10.0 ** (4.3 * rr)}})(
        float(np.clip((radius / 1.2) ** 0.8, 0.6, 1.4))),
    'count': "Family member",
    'storeys_label': "Periods",
    'test_order': 2,
}
SURFACE_FAMILY['SP_DASILVA_BATISTA'] = 'SINGLY'


# ==========================================================================
# Translation-invariant catenoid/Costa towers + CHM variants
# (appended catalog block; stinv_* engine in weierstrass)
# ==========================================================================
# The five remaining translation-invariant singly periodic surfaces from
# the minimalsurfaces.blog harvest, all with their notebook period
# constants re-extracted and machine-verified (engine block + gates in
# weierstrass; every member's per-period Euler characteristic is
# MEASURED against 2 - 2 genus - #ends):
#   * SP_CAT_HANDLE_G1   fence of catenoids with ONE extra handle per
#     period: genus 2, 2 catenoid ends, chi/period = -4.  Solved pairs
#     (a, b) from Singly_Catenoid_1Handle_g1.nb.
#   * SP_CAT_HANDLES_G3  fence of catenoids with TWO extra handles:
#     genus 3, 2 ends, chi/period = -6.  Solved triples (a, b, c) from
#     Singly_Catenoid_2Handles_g3.nb.
#   * SP_COSTA_TRANSINV  translation-invariant Costa I: genus 1, 4 ends
#     per period (2 catenoid-type + 2 flat annular wings),
#     chi/period = -4.  Solved triples (a, b, rho) from
#     Singly_TransInvCosta_I.nb, meshed on the Joukowski half-disk
#     chart.
#   * CHM12_PERIODIC     Callahan-Hoffman-Meeks CHM-(1,2): MEASURED
#     quotient genus 4 with 2 horizontal planar ends per period
#     (chi = -8) -- one handle more than CHM_PERIODIC's genus 3; the
#     constants are Newton-polished from the Singly_CHM_1_2.nb seed
#     (residual ~4e-6 vs the notebook's ~1.4e-3) and the strip chart
#     x = sqrt(b^2 + e^w) parallels the CHM-(1,1) build, with a
#     combinatorial 8-isometry weld.
#   * SP_SCREW_CHM       Weber's screw-motion CHM: the CHM-(1,1) tower
#     deformed so consecutive storeys are ROTATED, not just translated
#     (theta-function Gauss map on the rectangular tau-torus; solved
#     (u, v) per tau from Singly_ScrewMotion_CHM.nb).  Quotient genus 3
#     with 2 ends per screw period (chi = -6), the same topology as its
#     translational CHM-(1,1) limit -- measured.
# The harvest's plain translation-invariant catenoid
# (Singly_TransInvCatenoid.nb) is ALREADY shipped as SP_FENCE_CAT --
# identical Weierstrass data (rho = sqrt(a), dh = dz/z, a = 0.2,
# r1 = 6), so no separate row is added.
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) 83-114;
#   M. J. Callahan, D. Hoffman, W. H. Meeks III, "Embedded minimal
#     surfaces with an infinite number of ends", Invent. Math. 96
#     (1989) 459-505;
#   C. J. Costa (1984); D. Hoffman, W. H. Meeks III (1985);
#   M. Weber, https://minimalsurfaces.blog/ -- the harvested notebooks
#     (research/msblog_harvest/singly_periodic.json).

WE_SURFACES['SP_CAT_HANDLE_G1'] = {
    'label': "Catenoid Tower with Handle (genus 2)",
    'family': 'SINGLY',
    'mesher': we.stinv_g1_mesh,
    # count slider walks the solved neck moduli; radius digs the
    # catenoid funnels deeper (bigger r1 = deeper trim)
    'p_from': lambda order, radius: {
        'a': (0.1, 0.2, 0.3, 0.5)[int(np.clip(order, 1, 4)) - 1],
        'r1': float(np.clip(3.0 * (radius / 1.2) ** 1.5, 1.6, 9.0))},
    'count': "Neck modulus (a)",
    'storeys_label': "Periods",
    'test_order': 4,
}
SURFACE_FAMILY['SP_CAT_HANDLE_G1'] = 'SINGLY'

WE_SURFACES['SP_CAT_HANDLES_G3'] = {
    'label': "Catenoid Tower with 2 Handles (genus 3)",
    'family': 'SINGLY',
    'mesher': we.stinv_g3_mesh,
    'p_from': lambda order, radius: {
        'a': (0.06, 0.08, 0.1, 0.2, 0.4, 0.6)[
            int(np.clip(order, 1, 6)) - 1],
        'r1': float(np.clip(3.5 * (radius / 1.2) ** 1.5, 1.6, 10.0))},
    'count': "Neck modulus (a)",
    'storeys_label': "Periods",
    'test_order': 3,
}
SURFACE_FAMILY['SP_CAT_HANDLES_G3'] = 'SINGLY'

WE_SURFACES['SP_COSTA_TRANSINV'] = {
    'label': "Translation-Invariant Costa",
    'family': 'SINGLY',
    'mesher': we.stinv_costa_mesh,
    # count walks the solved (a, b, rho) family (a -> -1 squeezes the
    # wings together); radius extends the flat wings (smaller rmin)
    # and digs the catenoid funnels (smaller corner delta)
    'p_from': lambda order, radius: {
        'a': (-10.0, -5.0, -3.0, -2.0, -1.5, -1.1)[
            int(np.clip(order, 1, 6)) - 1],
        'rmin': float(np.clip(0.02 * (1.2 / max(radius, 0.2)) ** 1.2,
                              0.008, 0.06)),
        'delta': float(np.clip(0.12 * (1.2 / max(radius, 0.2)) ** 0.7,
                               0.06, 0.2))},
    'count': "Wing modulus (a)",
    'storeys_label': "Periods",
    'test_order': 1,
}
SURFACE_FAMILY['SP_COSTA_TRANSINV'] = 'SINGLY'

WE_SURFACES['CHM12_PERIODIC'] = {
    'label': "Callahan-Hoffman-Meeks CHM-(1,2) (genus 4)",
    'family': 'SINGLY',
    'mesher': we.stinv_chm12_mesh,
    # radius slides the two conformal end trims (wider flat shelves)
    'p_from': lambda order, radius: {
        'umin': float(np.clip(-4.0 - 2.0 * math.log(
            max(radius, 0.2) / 1.2), -6.5, -2.5)),
        'umax': float(np.clip(4.0 + 2.0 * math.log(
            max(radius, 0.2) / 1.2), 2.5, 6.5))},
    'storeys_label': "Periods",
    'test_order': 1,
}
SURFACE_FAMILY['CHM12_PERIODIC'] = 'SINGLY'

WE_SURFACES['SP_SCREW_CHM'] = {
    'label': "Screw-Motion CHM Tower",
    'family': 'SINGLY',
    'mesher': we.stinv_screw_mesh,
    # count walks the solved tau family (bigger tau = taller storey and
    # stronger twist); radius digs the two trimmed ends deeper
    'p_from': lambda order, radius: {
        'timag': (0.6, 0.7, 0.8, 0.9, 1.0, 1.05)[
            int(np.clip(order, 1, 6)) - 1],
        'delta': float(np.clip(0.10 * (1.2 / max(radius, 0.2)) ** 0.7,
                               0.05, 0.16))},
    'count': "Torus modulus (tau)",
    'storeys_label': "Periods",
    'test_order': 3,
}
SURFACE_FAMILY['SP_SCREW_CHM'] = 'SINGLY'


# ==========================================================================
# SFK TAIL (appended catalog block) -- Fischer-Koch towers and
# annular-ended genus-1 tori (singly periodic; engine: we.sfk_* block)
# ==========================================================================
# The singly periodic members that were deferred pending per-notebook
# constant extraction, now with their solved constants wired in:
#
#   * SP_FISCHER_KOCH  translation-invariant Fischer-Koch surface
#     (theta data on the rhombic torus C/(1, e^(i pi t0)); the
#     notebook's FindRoot literals (b0, t0) for k = 3, 5).  Genus 1
#     with 2k Scherk wing ends per period (measured chi/period = -2k);
#     the vertical rise Re Int dh = -1/k over the half lattice cycle
#     closes to ~1e-14 (the notebook's own period equation).  Only the
#     odd-k members ship: for even k two 2-fold axes of the orbit
#     coincide exactly (the classical embedded-iff-k-odd restriction)
#     and the mesh would self-intersect along them -- BACKLOG.md.
#   * SP_FK_FREESE     Fischer-Koch-Freese twist family (after
#     R. Freese): the mu-power theta deformation, invariant under the
#     period SCREW Rz(-2 pi mu) + (0,0,-2).  Gauss map multivalued
#     (G = B H^mu); the engine carries a continuous branch of log H
#     across the chart.  k = 3 with mu from the notebook's soln3
#     continuation table; mu -> 0 is SP_FISCHER_KOCH.  chi/period = -6
#     (measured, screw-wrapped quotient), genus 1, 6 wing ends.
#   * SP_1CAT_2ANN     singly periodic genus-1 torus with 1 catenoid +
#     2 annular ends: sqrt data branched at {0, a, 1, b}, dh = dz/z.
#     TWO period conditions Re Int_a^1 om1 = Re Int_1^b om2 = 0 are
#     solved by Newton from the notebook's harvested seeds (the
#     parallel-end member a = 0.2574798..., rho = 1 is reproduced to
#     ~1e-9).  MEASURED quotient chi = -3 with 3 end rims -- genus 1,
#     matching the harvest's expected chi.
#   * SP_2ENN_2ANN     translation-invariant torus with 2 Enneper + 2
#     annular ends: rational data, period problem closed in CLOSED
#     FORM (a = b/(1 - b^2 + sqrt(1 - b^2)) cancels the om2 residues).
#     MEASURED quotient chi = -2 with 4 end rims -- the parametrizing
#     surface is the 4-punctured SPHERE (both g and dh rational in z),
#     so the quotient genus is 0; the harvest's "genus 1" annotation
#     contradicts its own rational data, exactly like the shipped
#     sibling SP_ENNEPER_3ANN (see the weierstrass sptail note).
#
# Deferred (BACKLOG.md): hackman_surfaces (Weierstrass-sigma data with
# a transcendental Bonnet phase and its own torus uniformization),
# even-k Fischer-Koch / Freese members (self-intersecting), and the
# flipped-layout Freese branch mu < -0.05 at k = 4.
#
# References:
#   W. Fischer, E. Koch, "Spanning minimal surfaces", Phil. Trans. R.
#     Soc. Lond. A 354 (1996) 2105-2142;
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988);
#   M. Weber, https://minimalsurfaces.blog/ -- notebooks "Fischer-Koch
#     Translational", "Fischer-Koch-Freese" (after R. Freese), "Singly
#     Periodic Torus with One Catenoid and Two Annular Ends",
#     "Translation Invariant Torus with Two Enneper and Two Annular
#     Ends" (research/msblog_harvest/singly_periodic.json);
#   B. C. Carlson, Numer. Algorithms 10 (1995) 13-26 (R_F).

WE_SURFACES['SP_FISCHER_KOCH'] = {
    'label': "Fischer-Koch Tower (translation-invariant)",
    'family': 'SINGLY',
    'mesher': we.sfk_fkt_mesh,
    # count slider walks the odd wing number k = 3, 5; radius digs the
    # Scherk wing trims deeper (more negative strip rmin)
    'p_from': lambda order, radius: {
        'k': (3, 5)[int(np.clip(order, 1, 2)) - 1],
        'rmin': float(np.clip(-3.5 * (radius / 1.2) ** 0.8,
                              -6.0, -2.0))},
    'count': "Wings k (odd: 3, 5)",
    'storeys_label': "Periods",
    'test_order': 1,
}
SURFACE_FAMILY['SP_FISCHER_KOCH'] = 'SINGLY'

_SFK_FREESE_MUS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

WE_SURFACES['SP_FK_FREESE'] = {
    'label': "Fischer-Koch-Freese (twisted)",
    'family': 'SINGLY',
    'mesher': we.sfk_fkf_mesh,
    # count slider walks the screw twist mu (k = 3 fixed); radius digs
    # the wing trims deeper
    'p_from': lambda order, radius: {
        'mu': _SFK_FREESE_MUS[int(np.clip(order, 1, 6)) - 1],
        'rmin': float(np.clip(-3.5 * (radius / 1.2) ** 0.8,
                              -6.0, -2.0))},
    'count': "Twist mu (x 0.05)",
    'storeys_label': "Periods",
    'test_order': 3,
}
SURFACE_FAMILY['SP_FK_FREESE'] = 'SINGLY'

_SFK_C1A2_AS = (0.47, 0.40, 0.30, 0.25747983928707496, 0.10, 0.02)

WE_SURFACES['SP_1CAT_2ANN'] = {
    'label': "Torus with Catenoid + 2 Annular Ends",
    'family': 'SINGLY',
    'mesher': we.sfk_c1a2_mesh,
    # count slider walks the neck modulus a (order 4 = the parallel-
    # end member, rho = 1); radius slides the annular end reach
    'p_from': lambda order, radius: {
        'a': _SFK_C1A2_AS[int(np.clip(order, 1, 6)) - 1],
        'rmin': max(0.02, _SFK_C1A2_AS[
            int(np.clip(order, 1, 6)) - 1] / 5.0),
        'rmax': float(np.clip(12.0 * (radius / 1.2) ** 1.5,
                              6.0, 40.0))},
    'count': "Neck modulus (a)",
    'storeys_label': "Periods",
    'test_order': 4,
}
SURFACE_FAMILY['SP_1CAT_2ANN'] = 'SINGLY'

_SFK_E2A2_BS = (0.35, 0.45, 0.50, 0.55, 0.65, 0.75)

WE_SURFACES['SP_2ENN_2ANN'] = {
    'label': "Torus with 2 Enneper + 2 Annular Ends",
    'family': 'SINGLY',
    'mesher': we.sfk_e2a2_mesh,
    # count slider walks the modulus b; radius slides the Enneper end
    # trim (smaller rmin = wider flare)
    'p_from': lambda order, radius: {
        'b': _SFK_E2A2_BS[int(np.clip(order, 1, 6)) - 1],
        'rmin': float(np.clip(0.14 * (1.2 / max(radius, 0.2)) ** 0.8,
                              0.06, 0.30))},
    'count': "Modulus (b)",
    'storeys_label': "Periods",
    'test_order': 3,
}
SURFACE_FAMILY['SP_2ENN_2ANN'] = 'SINGLY'


def _selftest():
    # standalone catalog tests: build every row through the meshing
    # pipeline, then the engine-level QA gates (period closure,
    # translation structure, Bjorling seed reproduction)
    #
    # Imported here rather than at module scope on purpose: `parametric`
    # imports THIS module to register the catalog, so a top-level import
    # would close a cycle.  By the time a self-test runs, `parametric` is
    # fully initialized.
    from . import parametric as tk
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
    # Jeener's flower: the row is given as Weierstrass data, but the
    # source states the immersion in CLOSED FORM, so check the two
    # against each other rather than merely confirming the row builds.
    # This is the only gate that would catch the (g, dh) <-> (f, g)
    # conversion being off by a power -- dh = 2 g f, not f -- which
    # produces a perfectly good minimal surface that is not this one.
    spec = WE_SURFACES['JEENER']
    p = spec['p_from'](spec['test_order'], 1.2)
    phi = we._phi_fn(spec, p, 0.0)
    zr = np.random.default_rng(20260821)
    zs = (zr.uniform(-1.4, 1.4, 300) + 1j * zr.uniform(-1.4, 1.4, 300))
    zs = zs[np.abs(zs) > 0.05]
    got = np.asarray(phi(zs))
    # d/dw of (Re(w^3/3 - w^5/5), Re(i(w^3/3 + w^5/5)), Re(w^4/2))
    want = np.stack([zs ** 2 - zs ** 4,
                     1j * (zs ** 2 + zs ** 4),
                     2.0 * zs ** 3], axis=-1)
    jd = float(np.max(np.abs(got - want))) / float(np.max(np.abs(want)))
    # and phi.phi = 0 is what makes it conformal, hence minimal
    conf = float(np.max(np.abs((want * want).sum(-1))))
    jgood = jd < 1e-12 and conf < 1e-9
    ok &= jgood
    print(f"jeener flower: phi vs the closed form Re(w^3/3 - w^5/5) "
          f"etc. = {jd:.2e}, |sum phi_i^2| = {conf:.2e} "
          f"{'OK' if jgood else 'FAIL'}")

    # Karcher unequal-wing tower: the period-closure gate above only checks
    # alpha = 0 (the p_from default).  Sweep the whole alpha range at several
    # n, integrating the real horizontal periods around every end by contour,
    # and rebuild the mesh at a strong alpha -- the closure must hold and the
    # unit must still fit / stay manifold as the wings go unequal.
    spec = WE_SURFACES['SADDLE_TOWER_A']
    aworst = 0.0
    for nn in (2, 3, 4, 5):
        for alpha in np.linspace(0.0, math.pi / 2, 7):
            p = {'n': nn, 'alpha': float(alpha)}
            for (zc, r) in spec['cycles'](p):
                for comp in (0, 1):        # horizontal periods must vanish
                    I = we.period_integral(
                        lambda z, c=comp: _atower_phi(z, p)[c], zc, r, r)
                    aworst = max(aworst, abs(I.real))
    agood = aworst < 1e-6
    ok &= agood
    print(f"alpha-tower closure (n=2..5, alpha=0..pi/2): "
          f"max|Re oint phi_h| = {aworst:.2e} {'OK' if agood else 'FAIL'}")
    amax = 0.0
    for alpha in (0.0, math.pi / 6, math.pi / 3, math.pi / 2):
        for order in (1, 3):               # n = 2 and n = 4
            V, Q = tk.build_parametric('SADDLE_TOWER_A', 60, 60, order,
                                       1.2, 1.0, float(alpha))
            lo, hi = V.min(0), V.max(0)
            cen = float(np.max(np.abs(0.5 * (lo + hi))))
            ext = float(np.max(hi - lo))
            ec = {}
            for f in Q:
                for t in range(len(f)):
                    a, b = f[t], f[(t + 1) % len(f)]
                    e = (a, b) if a < b else (b, a)
                    ec[e] = ec.get(e, 0) + 1
            nm = sum(1 for c in ec.values() if c > 2)
            g2 = (bool(np.all(np.isfinite(V))) and len(Q) > 100
                  and cen < 1e-6 and abs(ext - 2.0) < 1e-6 and nm == 0)
            ok &= g2
            amax = max(amax, 0.0 if g2 else 1.0)
            print(f"alpha-tower mesh a={alpha:.3f} n={order + 1}: "
                  f"{len(V):5d}v {len(Q):5d}f fit[|c|={cen:.1e} "
                  f"ext={ext:.4f}] nonman={nm} {'OK' if g2 else 'FAIL'}")
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
    # Higher-genus Chen-Gackstatter: the full watertight gate through the
    # toolkit pipeline -- exact Euler characteristic chi = 1 - 2g (2 - 2g
    # for the closed surface, minus the one trimmed Enneper end), zero
    # non-manifold edges, ONE boundary loop, one component, 2 m fit, and
    # a finite per-corner UV chart.
    for gord, gg in ((2, 2), (4, 4), (5, 5)):
        out = tk.build_parametric('CG_HIGHER', 60, 60, gord, 1.2, 1.0,
                                  with_uv=True)
        Vg, Qg, uvg = out
        ecg = {}
        for f in Qg:
            m = len(f)
            for t in range(m):
                a, b = f[t], f[(t + 1) % m]
                e = (a, b) if a < b else (b, a)
                ecg[e] = ecg.get(e, 0) + 1
        chi = len(Vg) - len(ecg) + len(Qg)
        nonman = sum(1 for c in ecg.values() if c > 2)
        bed = [e for e, c in ecg.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a, b in bed:
            ra, rb = bfind(a), bfind(b)
            if ra != rb:
                par[ra] = rb
        nloops = len({bfind(a) for a, b in bed})
        parc = list(range(len(Vg)))

        def cfind(a):
            while parc[a] != a:
                parc[a] = parc[parc[a]]
                a = parc[a]
            return a

        for f in Qg:
            for i in range(1, len(f)):
                ra, rb = cfind(f[0]), cfind(f[i])
                if ra != rb:
                    parc[ra] = rb
        ncomp = len({cfind(f[0]) for f in Qg})
        lo, hi = Vg.min(0), Vg.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        uv_ok = (uvg is not None and len(uvg) == sum(len(f) for f in Qg)
                 and bool(np.all(np.isfinite(uvg))))
        good = (chi == 1 - 2 * gg and nonman == 0 and nloops == 1
                and ncomp == 1 and cen < 1e-6 and abs(ext - 2.0) < 1e-6
                and uv_ok and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"CG higher g{gg}: {len(Vg):6d}v {len(Qg):6d}f chi={chi} "
              f"(want {1 - 2 * gg}) nonman={nonman} loops={nloops} "
              f"ncomp={ncomp} fit[|c|={cen:.1e} ext={ext:.4f}] "
              f"uv={uv_ok} {'OK' if good else 'FAIL'}")
    # Genus-one helicoid through the toolkit pipeline: a stack of S
    # translational periods must be one connected manifold surface of
    # genus exactly S (chi = 2 - 2S - boundary_loops), fit to 2 m.
    for S in (1, 2):
        Vg, Qg = tk.build_parametric('GENUS1_HELICOID', 60, 60, S, 1.2,
                                     1.0)
        ecg = {}
        for f in Qg:
            m = len(f)
            for t in range(m):
                a, b = f[t], f[(t + 1) % m]
                e = (a, b) if a < b else (b, a)
                ecg[e] = ecg.get(e, 0) + 1
        chi = len(Vg) - len(ecg) + len(Qg)
        nonman = sum(1 for c in ecg.values() if c > 2)
        bed = [e for e, c in ecg.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a, b in bed:
            ra, rb = bfind(a), bfind(b)
            if ra != rb:
                par[ra] = rb
        nloops = len({bfind(a) for a, b in bed})
        genus = (2 - chi - nloops) / 2.0
        lo, hi = Vg.min(0), Vg.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        good = (genus == S and nonman == 0 and cen < 1e-6
                and abs(ext - 2.0) < 1e-6
                and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"g1-helicoid S={S}: {len(Vg):6d}v {len(Qg):6d}f "
              f"chi={chi} loops={nloops} genus={genus:.0f} (want {S}) "
              f"nonman={nonman} fit[|c|={cen:.1e} ext={ext:.4f}] "
              f"{'OK' if good else 'FAIL'}")
    # Symmetrized Chen-Gackstatter towers: the same full watertight gate
    # through the toolkit pipeline -- exact chi = 1 - 2g (one trimmed
    # Enneper end), manifold, one loop, one component, 2 m fit, UV chart.
    # SYMM_CG order k has genus k - 1 (k = 4 -> genus 3); SYMM_CG_G2N
    # genus 2(k-1); SYMM_CG_G3K genus 3(k-1).
    for skey, sord, sgen in (('SYMM_CG', 2, 1), ('SYMM_CG', 4, 3),
                             ('SYMM_CG', 6, 5), ('SYMM_CG_G2N', 3, 4),
                             ('SYMM_CG_G3K', 2, 3)):
        Vg, Qg, uvg = tk.build_parametric(skey, 60, 60, sord, 1.2, 1.0,
                                          with_uv=True)
        ecg = {}
        for f in Qg:
            m = len(f)
            for t in range(m):
                a, b = f[t], f[(t + 1) % m]
                e = (a, b) if a < b else (b, a)
                ecg[e] = ecg.get(e, 0) + 1
        chi = len(Vg) - len(ecg) + len(Qg)
        nonman = sum(1 for c in ecg.values() if c > 2)
        bed = [e for e, c in ecg.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a, b in bed:
            ra, rb = bfind(a), bfind(b)
            if ra != rb:
                par[ra] = rb
        nloops = len({bfind(a) for a, b in bed})
        parc = list(range(len(Vg)))

        def cfind(a):
            while parc[a] != a:
                parc[a] = parc[parc[a]]
                a = parc[a]
            return a

        for f in Qg:
            for i in range(1, len(f)):
                ra, rb = cfind(f[0]), cfind(f[i])
                if ra != rb:
                    parc[ra] = rb
        ncomp = len({cfind(f[0]) for f in Qg})
        lo, hi = Vg.min(0), Vg.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        uv_ok = (uvg is not None and len(uvg) == sum(len(f) for f in Qg)
                 and bool(np.all(np.isfinite(uvg))))
        good = (chi == 1 - 2 * sgen and nonman == 0 and nloops == 1
                and ncomp == 1 and cen < 1e-6 and abs(ext - 2.0) < 1e-6
                and uv_ok and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"symm-CG {skey:12s} k={sord} (genus {sgen}): {len(Vg):6d}v "
              f"{len(Qg):6d}f chi={chi} (want {1 - 2 * sgen}) "
              f"nonman={nonman} loops={nloops} ncomp={ncomp} "
              f"fit[|c|={cen:.1e} ext={ext:.4f}] uv={uv_ok} "
              f"{'OK' if good else 'FAIL'}")
    # Singly periodic Callahan-Hoffman-Meeks: full pipeline gate per
    # stacked period count S.  The quotient by one translation is genus
    # 2k + 1 = 3 with two planar ends, so S welded periods mesh at
    # exactly chi = -6 S with 2 S end rims + 2 outer horizontal cuts,
    # edge-manifold, one component, 2 m fit.
    for S in (1, 2):
        Vp, Qp = tk.build_parametric('CHM_PERIODIC', 60, 60, 1, 1.2, 1.0,
                                     cells=(S, 1))
        ecp = {}
        for f in Qp:
            m = len(f)
            for tq in range(m):
                a, b = f[tq], f[(tq + 1) % m]
                e = (a, b) if a < b else (b, a)
                ecp[e] = ecp.get(e, 0) + 1
        chi = len(Vp) - len(ecp) + len(Qp)
        nonman = sum(1 for c in ecp.values() if c > 2)
        bed = [e for e, c in ecp.items() if c == 1]
        parb = {}

        def pfind(x):
            parb.setdefault(x, x)
            while parb[x] != x:
                parb[x] = parb[parb[x]]
                x = parb[x]
            return x

        for a, b in bed:
            ra, rb = pfind(a), pfind(b)
            if ra != rb:
                parb[ra] = rb
        nloops = len({pfind(a) for a, b in bed})
        parc = list(range(len(Vp)))

        def cfind2(a):
            while parc[a] != a:
                parc[a] = parc[parc[a]]
                a = parc[a]
            return a

        for f in Qp:
            for i in range(1, len(f)):
                ra, rb = cfind2(f[0]), cfind2(f[i])
                if ra != rb:
                    parc[ra] = rb
        ncomp = len({cfind2(f[0]) for f in Qp})
        lo, hi = Vp.min(0), Vp.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        good = (chi == -6 * S and nonman == 0 and nloops == 2 * S + 2
                and ncomp == 1 and cen < 1e-6 and abs(ext - 2.0) < 1e-6
                and bool(np.all(np.isfinite(Vp))))
        ok &= good
        print(f"CHM periodic S={S}: {len(Vp):6d}v {len(Qp):6d}f chi={chi} "
              f"(want {-6 * S}) nonman={nonman} loops={nloops} "
              f"(want {2 * S + 2}) ncomp={ncomp} fit[|c|={cen:.1e} "
              f"ext={ext:.4f}] {'OK' if good else 'FAIL'}")
    # Catenoid-Enneper / Costa-Wohlgemuth / Wohlgemuth: full pipeline
    # watertight gates -- exact chi = 2 - 2 genus - (open end rims),
    # edge-manifold, one component, 2 m fit, finite UV chart.  The
    # engine-level period/null gates live in weierstrass._selftest().
    for wkey, word, wgen, wrim in (('CATENOID_ENNEPER', 2, 2, 2),
                                   ('CATENOID_ENNEPER', 3, 3, 2),
                                   ('CATENOID_ENNEPER', 4, 4, 2),
                                   ('COSTA_WOHLGEMUTH', 2, 2, 4),
                                   ('COSTA_WOHLGEMUTH', 3, 4, 4),
                                   ('WOHLGEMUTH_G3', 1, 3, 4)):
        Vg, Qg, uvg = tk.build_parametric(wkey, 60, 60, word, 1.2, 1.0,
                                          with_uv=True)
        ecg = {}
        for f in Qg:
            m = len(f)
            for t in range(m):
                a, b = f[t], f[(t + 1) % m]
                e = (a, b) if a < b else (b, a)
                ecg[e] = ecg.get(e, 0) + 1
        chi = len(Vg) - len(ecg) + len(Qg)
        nonman = sum(1 for c in ecg.values() if c > 2)
        bed = [e for e, c in ecg.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a, b in bed:
            ra, rb = bfind(a), bfind(b)
            if ra != rb:
                par[ra] = rb
        nloops = len({bfind(a) for a, b in bed})
        parc = list(range(len(Vg)))

        def cfind(a):
            while parc[a] != a:
                parc[a] = parc[parc[a]]
                a = parc[a]
            return a

        for f in Qg:
            for i in range(1, len(f)):
                ra, rb = cfind(f[0]), cfind(f[i])
                if ra != rb:
                    parc[ra] = rb
        ncomp = len({cfind(f[0]) for f in Qg})
        lo, hi = Vg.min(0), Vg.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        uv_ok = (uvg is not None and len(uvg) == sum(len(f) for f in Qg)
                 and bool(np.all(np.isfinite(uvg))))
        want = 2 - 2 * wgen - wrim
        good = (chi == want and nonman == 0 and nloops == wrim
                and ncomp == 1 and cen < 1e-6 and abs(ext - 2.0) < 1e-6
                and uv_ok and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"{wkey} n={word} (genus {wgen}): {len(Vg):6d}v "
              f"{len(Qg):6d}f chi={chi} (want {want}) nonman={nonman} "
              f"loops={nloops} (want {wrim}) ncomp={ncomp} "
              f"fit[|c|={cen:.1e} ext={ext:.4f}] uv={uv_ok} "
              f"{'OK' if good else 'FAIL'}")
    # Doubly periodic KMR + Wei rows through the FULL toolkit pipeline
    # with a 2 x 2 lattice tiling (the operator's Cells U x V path):
    # one connected component (the tiles genuinely weld -- the Scherk
    # lesson), edge-manifold, 2 m fit, per-corner conformal UV.  (The
    # quotient topology chi = 2 - 2g - 4 is gated in weierstrass.)
    for dkey, dord in (('KMR_DOUBLY', 3), ('KMR3_DOUBLY', 1),
                       ('WEI_DOUBLY', 1)):
        Vd, Qd, uvd = tk.build_parametric(dkey, 48, 48, dord, 1.2, 1.0,
                                          with_uv=True, cells=(2, 2))
        ecd = {}
        for f in Qd:
            m = len(f)
            for tq in range(m):
                a, b = f[tq], f[(tq + 1) % m]
                e = (a, b) if a < b else (b, a)
                ecd[e] = ecd.get(e, 0) + 1
        nonman = sum(1 for c in ecd.values() if c > 2)
        parc = list(range(len(Vd)))

        def dfind(a):
            while parc[a] != a:
                parc[a] = parc[parc[a]]
                a = parc[a]
            return a

        for f in Qd:
            for i in range(1, len(f)):
                ra, rb = dfind(f[0]), dfind(f[i])
                if ra != rb:
                    parc[ra] = rb
        ncomp = len({dfind(f[0]) for f in Qd})
        lo, hi = Vd.min(0), Vd.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        uv_ok = (uvd is not None and len(uvd) == sum(len(f) for f in Qd)
                 and bool(np.all(np.isfinite(uvd))))
        good = (nonman == 0 and ncomp == 1 and cen < 1e-6
                and abs(ext - 2.0) < 1e-6 and uv_ok and len(Qd) > 1000
                and bool(np.all(np.isfinite(Vd))))
        ok &= good
        print(f"doubly {dkey:12s} 2x2: {len(Vd):6d}v {len(Qd):6d}f "
              f"nonman={nonman} ncomp={ncomp} fit[|c|={cen:.1e} "
              f"ext={ext:.4f}] uv={uv_ok} {'OK' if good else 'FAIL'}")
    # Singly periodic long tail: full pipeline gates.  Each mesher row is
    # built through build_parametric at S = 1 and S = 2 stacked periods;
    # the measured chi(2) - chi(1) must equal the quotient Euler
    # characteristic 2 - 2 genus - #ends (the theorem check), and every
    # stack must be edge-manifold, one component and fit the 2 m cube.
    # (SP_PERIODIC_ENNEPER is one continuous grid strip -- chi = 1 at
    # every turn count.)
    for skey, sord, dchi in (('SP_SIX_SCHERK', 2, -4),
                             ('SP_ALT_FENCE', 3, -2),
                             ('SP_FENCE_CAT', 3, -2),
                             ('SP_HELICOIDAL_SCHERK', 3, -6),
                             ('SP_ENNEPER_3ANN', 1, -2),
                             ('SP_PERIODIC_ENNEPER', 1, 0),
                             # sscherk block: higher-genus towers --
                             # dchi per period = 2 - 2 genus - #ends
                             ('SP_SIX_SCHERK_G1', 4, -6),
                             ('SP_COSTA_SCHERK_G1', 2, -6),
                             ('SP_EIGHT_SCHERK_G2', 3, -10),
                             ('SP_DASILVA_BATISTA', 2, -10),
                             # SFK tail: Fischer-Koch towers + annular
                             # tori (dchi = quotient chi = 2 - 2g - E)
                             ('SP_FISCHER_KOCH', 1, -6),
                             ('SP_FK_FREESE', 3, -6),
                             ('SP_1CAT_2ANN', 4, -3),
                             ('SP_2ENN_2ANN', 3, -2)):
        chis = []
        good = True
        for S in (1, 2):
            Vs, Qs = tk.build_parametric(skey, 48, 48, sord, 1.2, 1.0,
                                         cells=(S, 1))
            ecs = {}
            for f in Qs:
                m = len(f)
                for tq in range(m):
                    a, b = f[tq], f[(tq + 1) % m]
                    e = (a, b) if a < b else (b, a)
                    ecs[e] = ecs.get(e, 0) + 1
            chis.append(len(Vs) - len(ecs) + len(Qs))
            nonman = sum(1 for c in ecs.values() if c > 2)
            parc = list(range(len(Vs)))

            def sfind(a):
                while parc[a] != a:
                    parc[a] = parc[parc[a]]
                    a = parc[a]
                return a

            for f in Qs:
                for i in range(1, len(f)):
                    ra, rb = sfind(f[0]), sfind(f[i])
                    if ra != rb:
                        parc[ra] = rb
            ncomp = len({sfind(f[0]) for f in Qs})
            lo, hi = Vs.min(0), Vs.max(0)
            cen = float(np.max(np.abs(0.5 * (lo + hi))))
            ext = float(np.max(hi - lo))
            good &= (nonman == 0 and ncomp == 1 and cen < 1e-6
                     and abs(ext - 2.0) < 1e-6
                     and bool(np.all(np.isfinite(Vs))))
        good &= (chis[1] - chis[0] == dchi)
        ok &= good
        print(f"sptail {skey:22s}: chi {chis[0]}->{chis[1]} "
              f"(dchi want {dchi}) manifold/fit "
              f"{'OK' if good else 'FAIL'}")
    # Scherk-Enneper: the vertical periods around consecutive wing ends
    # alternate +-T exactly (the surface's screw structure); horizontal
    # periods are gated by the generic cycles test above.
    spec = WE_SURFACES['SP_SCHERK_ENNEPER']
    seworst, sealt = 0.0, 0.0
    for order in (1, 2, 3):
        p = spec['p_from'](order, 1.2)
        phi = we._phi_fn(spec, p, 0.0)
        kk = p['k']
        dzs = []
        for j in range(2 * kk):
            zc = np.exp(1j * math.pi * j / kk)
            rr = 0.4 * min(0.16, 0.5 * math.sin(math.pi / (2 * kk)))
            I = we.period_integral(lambda z: phi(z)[..., 2], zc, rr, rr)
            dzs.append(I.real)
        for j in range(2 * kk):
            sealt = max(sealt, abs(dzs[j] + dzs[(j + 1) % (2 * kk)]))
        seworst = max(seworst, -min(abs(d) for d in dzs))
        seworst = max(seworst, 0.0)
        semag = min(abs(d) for d in dzs)
    good = sealt < 1e-9 and semag > 0.05
    ok &= good
    print(f"sptail SP_SCHERK_ENNEPER periods: alternation "
          f"residual={sealt:.1e} |T|>={semag:.3f} "
          f"{'OK' if good else 'FAIL'}")
    # ---- STINV towers + CHM variants (appended gate block) -----------
    # engine-level gates (period residuals at the harvested constants,
    # translation-wrapped quotient topology) live in weierstrass'
    # _selftest(); here the full pipeline is gated at S = 1 and S = 2
    # stacked periods: measured chi(2) - chi(1) must equal the quotient
    # Euler characteristic 2 - 2 genus - #ends, and every stack must be
    # edge-manifold, one component and fit the 2 m cube.
    for skey, sord, dchi in (('SP_CAT_HANDLE_G1', 4, -4),
                             ('SP_CAT_HANDLES_G3', 3, -6),
                             ('SP_COSTA_TRANSINV', 1, -4),
                             ('CHM12_PERIODIC', 1, -8),
                             ('SP_SCREW_CHM', 3, -6)):
        chis = []
        good = True
        for S in (1, 2):
            Vs, Qs = tk.build_parametric(skey, 48, 48, sord, 1.2, 1.0,
                                         cells=(S, 1))
            ecs = {}
            for f in Qs:
                m = len(f)
                for tq in range(m):
                    a, b = f[tq], f[(tq + 1) % m]
                    e = (a, b) if a < b else (b, a)
                    ecs[e] = ecs.get(e, 0) + 1
            chis.append(len(Vs) - len(ecs) + len(Qs))
            nonman = sum(1 for c in ecs.values() if c > 2)
            parc = list(range(len(Vs)))

            def sfind(a):
                while parc[a] != a:
                    parc[a] = parc[parc[a]]
                    a = parc[a]
                return a

            for f in Qs:
                for i in range(1, len(f)):
                    ra, rb = sfind(f[0]), sfind(f[i])
                    if ra != rb:
                        parc[ra] = rb
            ncomp = len({sfind(f[0]) for f in Qs})
            lo, hi = Vs.min(0), Vs.max(0)
            cen = float(np.max(np.abs(0.5 * (lo + hi))))
            ext = float(np.max(hi - lo))
            good &= (nonman == 0 and ncomp == 1 and cen < 1e-6
                     and abs(ext - 2.0) < 1e-6
                     and bool(np.all(np.isfinite(Vs))))
        good &= (chis[1] - chis[0] == dchi)
        ok &= good
        print(f"stinv {skey:22s}: chi {chis[0]}->{chis[1]} "
              f"(dchi want {dchi}) manifold/fit "
              f"{'OK' if good else 'FAIL'}")
    # ---- SYMM/NONORIENT TAIL gates -----------------------------------
    # Kusner: the FULL residue (Re and Im) must vanish at every one of
    # the 2p planar ends -- the immersion is single-valued with no
    # period problem (p = 7 included, beyond the shipped test orders).
    for pp in (3, 5, 7):
        spec = WE_SURFACES['KUSNER_RP2']
        p = spec['p_from']((pp - 1) // 2, 1.2)
        phi = we._phi_fn(spec, p, 0.0)
        s = p['s']
        r_in = ((pp - s) / (pp - 1)) ** (1.0 / pp)
        ends = [r_in * np.exp(2j * math.pi * j / pp) for j in range(pp)]
        ends += [np.exp(1j * (2 * math.pi * j + math.pi) / pp) / r_in
                 for j in range(pp)]
        w = 0.0
        for zc in ends:
            for c in range(3):
                I = we.period_integral(lambda z, c=c: phi(z)[..., c],
                                       zc, 0.05, 0.05)
                w = max(w, abs(I))
        good = w < 1e-6
        ok &= good
        print(f"kusner residues p={pp}: max|oint phi| = {w:.2e} "
              f"{'OK' if good else 'FAIL'}")
    # Henneberg: the antipodal identity X(-1/conj z) = X(z) holds to
    # machine epsilon on the exact antiderivative (one-sidedness of the
    # underlying immersion, independent of any mesh).
    rngh = np.random.default_rng(7)
    zh = rngh.uniform(0.4, 2.4, 64) \
        * np.exp(1j * rngh.uniform(0.0, TAU, 64))
    Xa = np.stack(_symtail_henneberg_X(zh, {}), axis=-1)
    Xb = np.stack(_symtail_henneberg_X(-1.0 / np.conj(zh), {}), axis=-1)
    e = float(np.max(np.abs(Xa - Xb)))
    good = e < 1e-10
    ok &= good
    print(f"henneberg antipodal identity: max err = {e:.2e} "
          f"{'OK' if good else 'FAIL'}")
    # Antiprismatic k-noid: the numeric two-ring Lopez-Ros solve closes
    # every period for the whole slider range, and reproduces Weber's
    # harvested nn = 5, b = 0.2 constants (the M3_ANTI5 member).
    ah, rh = we.symtail_antiprism_constants(5, 0.2)
    e = max(abs(ah - 0.2748767946679093),
            abs(rh - 0.0015692436842339352))
    good = e < 1e-8
    ok &= good
    print(f"antiprism harvested check: a={ah:.12f} rho={rh:.6e} "
          f"err={e:.2e} {'OK' if good else 'FAIL'}")
    for nn in (3, 4, 5, 6, 7):
        try:
            aa, rr = we.symtail_antiprism_constants(nn, _SYMTAIL_AP_B[nn])
            print(f"antiprism nn={nn}: a={aa:.10f} rho={rr:.4e} OK")
        except ValueError as ex:
            ok = False
            print(f"antiprism nn={nn}: FAIL ({ex})")
    # Non-orientable meshes: exact Euler characteristic, boundary loop
    # count, edge-manifold, 2 m fit, finite UV -- and MEASURED
    # one-sidedness (orientation propagation meets a contradiction;
    # the orientation double cover is connected).
    for nkey, nord, wchi, wloops in (('HENNEBERG_RP2', 1, 0, 1),
                                     ('KUSNER_RP2', 1, -2, 3),
                                     ('KUSNER_RP2', 2, -4, 5),
                                     ('LOPEZ_KLEIN', 1, -1, 1)):
        Vn, Qn, uvn = tk.build_parametric(nkey, 60, 60, nord, 1.2, 1.0,
                                          with_uv=True)
        chi, nonman, nloops, one_sided = we.symtail_edge_stats(Vn, Qn)
        lo, hi = Vn.min(0), Vn.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        # non-degeneracy: a genuinely 3-D surface, not a collapsed
        # plane (a flat complex can still pass every topology gate)
        ext_min = float(np.min(hi - lo))
        uv_ok = (uvn is not None and len(uvn) == sum(len(f) for f in Qn)
                 and bool(np.all(np.isfinite(uvn))))
        good = (chi == wchi and nloops == wloops and nonman == 0
                and one_sided and cen < 1e-6 and abs(ext - 2.0) < 1e-6
                and ext_min > 0.3 and uv_ok
                and bool(np.all(np.isfinite(Vn))))
        ok &= good
        print(f"nonorient {nkey:13s} n={nord}: {len(Vn):6d}v "
              f"{len(Qn):6d}f chi={chi} (want {wchi}) loops={nloops} "
              f"(want {wloops}) nonman={nonman} one_sided={one_sided} "
              f"fit[|c|={cen:.1e} ext={ext:.4f} min={ext_min:.2f}] "
              f"uv={uv_ok} {'OK' if good else 'FAIL'}")
    # Symmetrization-tail rows: one connected component at every
    # slider order (the object-space clip must not shear off islands)
    for skey, sords in (('SYMM_FRIEM', (2, 3, 4, 5, 6)),
                        ('SYMM_DBLENN', (2, 3, 4, 5, 6)),
                        ('KNOID_ENN_ENDS', (3, 4, 5, 6, 7)),
                        ('ANTIPRISM_KNOID', (1, 2, 3, 4, 5))):
        for so in sords:
            Vs2, Qs2 = tk.build_parametric(skey, 60, 60, so, 1.2, 1.0)
            parc2 = list(range(len(Vs2)))

            def sfind(a):
                while parc2[a] != a:
                    parc2[a] = parc2[parc2[a]]
                    a = parc2[a]
                return a

            for f in Qs2:
                for i in range(1, len(f)):
                    ra, rb = sfind(f[0]), sfind(f[i])
                    if ra != rb:
                        parc2[ra] = rb
            nc = len({sfind(f[0]) for f in Qs2})
            good = nc == 1 and len(Qs2) > 500
            ok &= good
            if not good:
                print(f"symm-tail {skey} order {so}: ncomp={nc} FAIL")
        print(f"symm-tail {skey}: one component at orders {sords} OK")
    # Symmetrization skip list (documented duplicates -- kept as a
    # printed record, not a gate): higher-order Enneper == ENNEPER,
    # symm-Scherk == SCHERK_TOWER, Jorge-Meeks == KNOID, pyramidal ==
    # M3_PYR, bipyramidal == M3_BIPYR, prismatic == M3_PRISM,
    # symm-Costa == COSTA_HM (rho == chm_modulus, see the TODO note),
    # symm-Chen-Gackstatter == SYMM_CG towers, Enneper-n-catenoids ==
    # ENNEPER_NCAT.
    print("symm tail: shipped SYMM_FRIEM SYMM_DBLENN KNOID_ENN_ENDS "
          "ANTIPRISM_KNOID + nonorient HENNEBERG_RP2 KUSNER_RP2 "
          "LOPEZ_KLEIN; 9 index entries skipped as duplicates")
    print("\nRESULT:", "ALL OK" if ok else "FAILURES in zoo")
    assert ok
