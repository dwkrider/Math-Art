
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
# and references live in we_builders.cg_higher_mesh and the block above
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
# the two end punctures -- gated below and in we_builders).  Weierstrass
# data and the two solved period constants (a, rho) are harvested from
# M. Weber's CHM-(1,1) notebook (research/msblog_harvest/
# singly_periodic.json); the engine, verified branch/strip chart and the
# 16-isometry period assembly live in we_builders.chm_periodic_mesh and
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
# Hoffman-Weber-Wolf 2009) live in we_builders.genus1helicoid_mesh and
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
# we_builders.symmcg_mesh and the block above it:
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
# Doubly periodic KMR + Wei surfaces (appended catalog block)
# ==========================================================================
# Three doubly periodic four-ended surfaces on the reusable hyperelliptic
# tiler in we_builders (dperiodic_*): ONE conformal patch is integrated in
# exponential coordinates, its boundary arcs are snapped exactly onto
# their vertical mirror planes / straight lines, and the surface is the
# orbit of that patch under the reflections/rotations those arcs
# generate, tiled at the TRUE lattice vectors and welded seam-exactly.
# The Cells U / V counts of the periodic operator drive the 2-D lattice
# tiling.  Wall-constant closure (the period problem), quotient topology
# (chi = 2 - 2g - 4, four Scherk-type end rims), manifoldness and
# orientability are all measured in the we_builders self-tests.
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
# References (full citations in the we_builders engine block):
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
    # Doubly periodic KMR + Wei rows through the FULL toolkit pipeline
    # with a 2 x 2 lattice tiling (the operator's Cells U x V path):
    # one connected component (the tiles genuinely weld -- the Scherk
    # lesson), edge-manifold, 2 m fit, per-corner conformal UV.  (The
    # quotient topology chi = 2 - 2g - 4 is gated in we_builders.)
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
    print("\nRESULT:", "ALL OK" if ok else "FAILURES in zoo")
