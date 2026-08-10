
# Atomic & Molecular Orbital Generator for Blender
#
# Hydrogenic atomic orbitals and LCAO molecular orbitals, meshed as
# isosurfaces of the wavefunction with the marching-tetrahedra
# extractor from the sibling Minimal Surface Toolkit module.  The
# angular parts are the real spherical harmonics supplied by the
# sibling Spherical Harmonic module; this module adds the radial part
# that turns an angular lobe picture into an actual orbital.
#
#   psi_nlm(r, theta, phi) = R_nl(r) Y_l^m(theta, phi)
#   R_nl(r) = N e^(-rho/2) rho^l L_(n-l-1)^(2l+1)(rho),  rho = 2 zeta r / n
#
# with L the generalised Laguerre polynomials (three-term recurrence --
# Blender ships no scipy) and N the normalisation that makes
# integral R_nl^2 r^2 dr = 1.  Radial nodes (n-l-1 of them) and angular
# nodes (l of them) are therefore present and correct: 3s shows its two
# inner shells, 4f its unpinched lobes.
#
# ATOMIC mode draws one psi_nlm.  MOLECULAR mode draws a linear
# combination of atomic orbitals (LCAO),
#
#   psi_MO = sum_i c_i chi_i,   chi_i centred on nucleus i,
#
# covering the textbook diatomic sigma/pi/delta bonding and
# antibonding pairs, the sp/sp2/sp3 hybrids, a symmetry-adapted water
# picture and the benzene pi system in the Hueckel approximation.  An
# arbitrary combination can be typed in directly.
#
# IMPORTANT -- what these are and are not: every orbital here is a
# hydrogenic (or Slater-type, via the zeta exponent) function, and
# every molecular orbital is a fixed, symmetry-adapted combination of
# such functions.  They are the qualitative pictures of an inorganic
# chemistry textbook, NOT self-consistent Hartree-Fock or DFT orbitals;
# the coefficients are chosen by symmetry (or by Hueckel theory), not
# variationally optimised.  They are drawn for their shape.
#
# The isosurface level is chosen by ENCLOSED PROBABILITY rather than by
# a raw value of psi: the operator samples |psi|^2, sorts it and finds
# the contour enclosing the requested fraction of the electron density.
# A raw level spanning many orders of magnitude between 1s and 5g would
# otherwise make the operator unusable.
#
# Sign convention: the real harmonics carry the Condon-Shortley phase,
# so e.g. p_x = Y_1^1 is the negative of x/r.  A global sign on a basis
# function has no effect on the shape of an orbital, and the bonding /
# antibonding combinations below are written to be correct as given.
#
# Geometry only; materials and rendering are left to Blender.
#
# References:
# - Hydrogen wavefunctions: E. Schroedinger, "Quantisierung als
#   Eigenwertproblem (Erste Mitteilung)", Annalen der Physik 384(4),
#   1926, pp. 361-376.
# - Standard treatment of R_nl, the real orbital forms and the LCAO
#   method: L. Pauling and E. B. Wilson, "Introduction to Quantum
#   Mechanics with Applications to Chemistry", McGraw-Hill, 1935.
# - Effective nuclear charge / Slater-type exponents: J. C. Slater,
#   "Atomic Shielding Constants", Physical Review 36, 1930, pp. 57-64.
# - Molecular orbitals as linear combinations of atomic orbitals:
#   F. Hund, "Zur Deutung der Molekelspektren", Zeitschrift fuer Physik,
#   1927-1928 (series); R. S. Mulliken, "Electronic Structures of
#   Polyatomic Molecules and Valence", Physical Review, 1932.
# - The benzene pi system: E. Hueckel, "Quantentheoretische Beitraege
#   zum Benzolproblem", Zeitschrift fuer Physik 70, 1931.
# - Generalised Laguerre recurrence: M. Abramowitz and I. A. Stegun,
#   "Handbook of Mathematical Functions", Dover, 1965, chapter 22.

bl_info = {
    "name": "Atomic & Molecular Orbital Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Atomic & Molecular Orbital",
    "description": "Hydrogenic atomic orbitals and LCAO molecular "
                   "orbitals as sign-coloured probability isosurfaces",
    "category": "Add Mesh",
}

import math
import re

import numpy as np


def _toolkit():
    """The sibling Minimal Surface Toolkit supplies marching_tets."""
    try:
        from . import minimal_surface_toolkit as mst
    except ImportError:
        import minimal_surface_toolkit as mst
    return mst


def _harmonics():
    """The sibling Spherical Harmonic module supplies real_sph_harm."""
    try:
        from . import spherical_harmonic_generator as shg
    except ImportError:
        import spherical_harmonic_generator as shg
    return shg


# ==========================================================================
# Radial functions
# ==========================================================================

def laguerre(k, alpha, x):
    """Generalised Laguerre polynomial L_k^alpha(x) by the recurrence
    (k+1) L_(k+1)^a = (2k+1+a-x) L_k^a - (k+a) L_(k-1)^a."""
    k = int(k)
    x = np.asarray(x, dtype=float)
    if k < 0:
        raise ValueError(f"laguerre needs k >= 0, got {k}")
    lkm1 = np.zeros_like(x)
    lk = np.ones_like(x)
    for i in range(k):
        lkp1 = ((2.0 * i + 1.0 + alpha - x) * lk
                - (i + alpha) * lkm1) / (i + 1.0)
        lkm1, lk = lk, lkp1
    return lk


def radial(n, l, r, zeta=1.0):
    """Hydrogenic radial function R_nl(r) in atomic units (a0 = 1),
    normalised so that integral R_nl(r)^2 r^2 dr = 1.  `zeta` is the
    effective nuclear charge, which turns the same expression into a
    Slater-type basis function."""
    n, l = int(n), int(l)
    if not (0 <= l < n):
        raise ValueError(f"radial needs 0 <= l < n, got n={n}, l={l}")
    r = np.asarray(r, dtype=float)
    rho = 2.0 * zeta * r / n
    # log-space normalisation: (2 zeta/n)^3 (n-l-1)! / (2n (n+l)!)
    lognorm = 0.5 * (3.0 * math.log(2.0 * zeta / n)
                     + math.lgamma(n - l)
                     - math.log(2.0 * n) - math.lgamma(n + l + 1.0))
    # rho^l e^(-rho/2) evaluated together so a large rho cannot make
    # the two factors overflow and underflow into a NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        env = np.where(rho > 0.0,
                       np.exp(l * np.log(np.maximum(rho, 1e-300))
                              - 0.5 * rho),
                       1.0 if l == 0 else 0.0)
    return math.exp(lognorm) * env * laguerre(n - l - 1, 2 * l + 1, rho)


def radial_extent(n, l, zeta=1.0, frac=0.999):
    """Smallest R with integral_0^R R_nl^2 r^2 dr >= frac -- the box the
    orbital actually needs.  Found by bisection on a fine quadrature."""
    hi = 4.0 * n * n / max(zeta, 1e-6) + 10.0
    r = np.linspace(1e-6, hi, 20000)
    dens = radial(n, l, r, zeta) ** 2 * r * r
    cum = np.cumsum(dens) * (r[1] - r[0])
    cum /= cum[-1]
    idx = int(np.searchsorted(cum, frac))
    return float(r[min(idx, len(r) - 1)])


# ==========================================================================
# Real orbital naming
# ==========================================================================
# name -> (l, m) in the real-harmonic convention of the sibling module
# (m > 0 is the cos(m phi) partner, m < 0 the sin(|m| phi) partner).

_ANGULAR = {
    's': (0, 0),
    'pz': (1, 0), 'px': (1, 1), 'py': (1, -1),
    'dz2': (2, 0), 'dxz': (2, 1), 'dyz': (2, -1),
    'dx2y2': (2, 2), 'dxy': (2, -2),
    'fz3': (3, 0), 'fxz2': (3, 1), 'fyz2': (3, -1),
    'fzx2y2': (3, 2), 'fxyz': (3, -2),
    'fx3': (3, 3), 'fy3': (3, -3),
}

_ORB_RE = re.compile(r'^([1-9])([spdf][a-z0-9]*)$')


def parse_orbital(name):
    """'3dxy' -> (n, l, m).  Raises ValueError on anything else."""
    key = str(name).strip().lower().replace('-', '').replace('_', '')
    mt = _ORB_RE.match(key)
    if not mt:
        raise ValueError(f"cannot read orbital name {name!r} "
                         f"(expected e.g. 1s, 2pz, 3dxy, 4fz3)")
    n = int(mt.group(1))
    tail = mt.group(2)
    if tail == 's':
        l, m = 0, 0
    elif tail in _ANGULAR:
        l, m = _ANGULAR[tail]
    else:
        raise ValueError(f"unknown orbital {name!r}; known angular "
                         f"parts are {sorted(_ANGULAR)}")
    if l >= n:
        raise ValueError(f"orbital {name!r} is impossible: needs l < n")
    return n, l, m


def orbital_name(n, l, m):
    for key, lm in _ANGULAR.items():
        if lm == (l, m):
            return f"{n}{key}"
    return f"({n},{l},{m})"


# ==========================================================================
# Basis functions and fields
# ==========================================================================

class Basis(object):
    """One atomic orbital chi_i: quantum numbers, centre, coefficient
    and effective exponent."""

    __slots__ = ('n', 'l', 'm', 'centre', 'coeff', 'zeta', 'label')

    def __init__(self, name, centre=(0.0, 0.0, 0.0), coeff=1.0,
                 zeta=1.0):
        self.n, self.l, self.m = parse_orbital(name)
        self.centre = np.asarray(centre, dtype=float)
        self.coeff = float(coeff)
        self.zeta = float(zeta)
        self.label = name

    @classmethod
    def from_numbers(cls, n, l, m, centre=(0.0, 0.0, 0.0), coeff=1.0,
                     zeta=1.0, label=None):
        """Build straight from quantum numbers, bypassing the chemical
        name table (which stops at f)."""
        self = cls.__new__(cls)
        self.n, self.l = int(n), int(l)
        self.m = int(np.clip(int(m), -self.l, self.l))
        self.centre = np.asarray(centre, dtype=float)
        self.coeff = float(coeff)
        self.zeta = float(zeta)
        self.label = label or f"({n},{l},{m})"
        return self

    def evaluate(self, X, Y, Z):
        shg = _harmonics()
        dx = X - self.centre[0]
        dy = Y - self.centre[1]
        dz = Z - self.centre[2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        rs = np.maximum(r, 1e-12)          # the nucleus is a removable
        theta = np.arccos(np.clip(dz / rs, -1.0, 1.0))
        phi = np.arctan2(dy, dx)
        return (radial(self.n, self.l, r, self.zeta)
                * shg.real_sph_harm(self.l, self.m, theta, phi))


def evaluate_lcao(basis, X, Y, Z):
    """psi = sum_i c_i chi_i on the sample grid."""
    out = np.zeros(np.shape(X), dtype=float)
    for b in basis:
        out += b.coeff * b.evaluate(X, Y, Z)
    return out


def probability_levels(basis, box, probabilities, samples=64):
    """The |psi| contours enclosing each of `probabilities` of the
    electron density, plus the half-width of the box the OUTERMOST of
    them needs.  Returns (levels, tight_half).

    One sampling pass serves every level, which is what makes the
    nested-shell cloud mode cost no more to choose than a single
    surface does."""
    g = np.linspace(-box, box, int(samples))
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    psi = np.abs(evaluate_lcao(basis, X, Y, Z))
    p2 = (psi * psi).ravel()
    order = np.argsort(p2)[::-1]
    cum = np.cumsum(p2[order])
    if cum[-1] <= 0.0:
        return [0.0] * len(probabilities), box
    levels = []
    for p in probabilities:
        idx = int(np.searchsorted(cum, float(p) * cum[-1]))
        levels.append(float(np.sqrt(p2[order[min(idx,
                                                 len(order) - 1)]])))
    widest = min(levels)
    inside = psi >= widest
    if not np.any(inside):
        return levels, box
    reach = 0.0
    for axis in ((1, 2), (0, 2), (0, 1)):
        reach = max(reach,
                    float(np.max(np.abs(g[np.any(inside, axis=axis)]))))
    return levels, min(box, reach + 2.0 * (g[1] - g[0]))


def probability_level(basis, box, probability=0.90, samples=64):
    """The |psi| contour enclosing `probability` of the electron
    density, plus the half-width of the box that contour actually
    needs.  Returns (level, tight_half).

    Working in enclosed probability rather than a raw level is what
    lets one operator serve 1s and 5g, whose peak amplitudes differ by
    orders of magnitude.  Reporting the tight box matters just as much:
    an orbital's 99.9%-density radius is mostly empty space, and
    marching over it wastes the sample budget exactly where the radial
    and angular nodes need it -- at low resolution the thin gap at a
    node falls between samples and neighbouring lobes fuse."""
    g = np.linspace(-box, box, int(samples))
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    psi = np.abs(evaluate_lcao(basis, X, Y, Z))
    p2 = (psi * psi).ravel()
    order = np.argsort(p2)[::-1]
    cum = np.cumsum(p2[order])
    if cum[-1] <= 0.0:
        return 0.0, box
    idx = int(np.searchsorted(cum, float(probability) * cum[-1]))
    level = float(np.sqrt(p2[order[min(idx, len(order) - 1)]]))
    inside = psi >= level
    if not np.any(inside):
        return level, box
    # the contour lives inside this; one coarse cell of margin keeps
    # the surface from touching the sample box and opening up
    reach = float(np.max(np.abs(g[np.any(inside, axis=(1, 2))])))
    for axis in ((0, 2), (0, 1)):
        reach = max(reach,
                    float(np.max(np.abs(g[np.any(inside, axis=axis)]))))
    margin = 2.0 * (g[1] - g[0])
    return level, min(box, reach + margin)


# ==========================================================================
# Molecular presets
# ==========================================================================
# Every preset returns (basis, label, bonds) for a given bond length d;
# `bonds` are index pairs into the list of distinct nuclear centres.

_WATER_ANGLE = math.radians(104.5 / 2.0)
_BENZENE_CC = 2.63          # a0, ~1.39 Angstrom


def _diatomic(orb, sign, d, label):
    a = (0.0, 0.0, -0.5 * d)
    b = (0.0, 0.0, 0.5 * d)
    return [Basis(orb, a, 1.0), Basis(orb, b, float(sign))], label


def molecular_basis(preset, d=1.4, huckel_k=0):
    """Basis, human label and nuclear geometry for a molecular preset.
    Returns (basis, label, centres, bonds)."""
    if preset == 'SIGMA_1S':
        basis, label = _diatomic('1s', +1, d, "sigma 1s (bonding)")
    elif preset == 'SIGMA_STAR_1S':
        basis, label = _diatomic('1s', -1, d, "sigma* 1s (antibonding)")
    elif preset == 'SIGMA_2S':
        basis, label = _diatomic('2s', +1, d, "sigma 2s (bonding)")
    elif preset == 'SIGMA_STAR_2S':
        basis, label = _diatomic('2s', -1, d, "sigma* 2s (antibonding)")
    elif preset == 'SIGMA_2PZ':
        # the two +z lobes point at each other, so the BONDING
        # combination is the difference
        basis, label = _diatomic('2pz', -1, d, "sigma 2pz (bonding)")
    elif preset == 'SIGMA_STAR_2PZ':
        basis, label = _diatomic('2pz', +1, d,
                                 "sigma* 2pz (antibonding)")
    elif preset == 'PI_2PX':
        basis, label = _diatomic('2px', +1, d, "pi 2px (bonding)")
    elif preset == 'PI_STAR_2PX':
        basis, label = _diatomic('2px', -1, d, "pi* 2px (antibonding)")
    elif preset == 'DELTA_3DXY':
        basis, label = _diatomic('3dxy', +1, d, "delta 3dxy (bonding)")
    elif preset == 'DELTA_STAR_3DXY':
        basis, label = _diatomic('3dxy', -1, d,
                                 "delta* 3dxy (antibonding)")
    elif preset == 'SP':
        s = 1.0 / math.sqrt(2.0)
        basis = [Basis('2s', coeff=s), Basis('2pz', coeff=s)]
        label = "sp hybrid"
    elif preset == 'SP2':
        basis = [Basis('2s', coeff=1.0 / math.sqrt(3.0)),
                 Basis('2px', coeff=math.sqrt(2.0 / 3.0))]
        label = "sp2 hybrid"
    elif preset == 'SP3':
        h = 0.5
        basis = [Basis('2s', coeff=h), Basis('2px', coeff=h),
                 Basis('2py', coeff=h), Basis('2pz', coeff=h)]
        label = "sp3 hybrid"
    elif preset in ('WATER_BOND', 'WATER_LONE_PAIR'):
        # O at the origin, the HOH bisector along +z, H atoms in the
        # xz-plane at half-angle 52.25 degrees, r(OH) = 1.81 a0
        roh = 1.81
        h1 = (roh * math.sin(_WATER_ANGLE), 0.0,
              roh * math.cos(_WATER_ANGLE))
        h2 = (-roh * math.sin(_WATER_ANGLE), 0.0,
              roh * math.cos(_WATER_ANGLE))
        if preset == 'WATER_BOND':
            # the totally symmetric (a1) O-H bonding combination
            basis = [Basis('2s', coeff=0.55, zeta=2.25),
                     Basis('2pz', coeff=0.55, zeta=2.25),
                     Basis('1s', h1, 0.40), Basis('1s', h2, 0.40)]
            label = "H2O a1 bonding orbital"
        else:
            # the a1 lone pair: the s-p mix pointing away from the H's
            basis = [Basis('2s', coeff=0.71, zeta=2.25),
                     Basis('2pz', coeff=-0.71, zeta=2.25)]
            label = "H2O a1 lone pair"
        centres = [(0.0, 0.0, 0.0), h1, h2]
        return basis, label, centres, [(0, 1), (0, 2)]
    elif preset == 'BENZENE_PI':
        k = int(huckel_k) % 6
        cen = [(_BENZENE_CC * math.cos(j * math.pi / 3.0),
                _BENZENE_CC * math.sin(j * math.pi / 3.0), 0.0)
               for j in range(6)]
        cs = huckel_benzene(k)
        basis = [Basis('2pz', cen[j], cs[j], zeta=1.6)
                 for j in range(6)]
        energy = 2.0 * math.cos(math.pi * ((k + 1) // 2) / 3.0)
        label = (f"benzene pi MO {k} (E = alpha + {energy:+.2f} beta)")
        bonds = [(j, (j + 1) % 6) for j in range(6)]
        return basis, label, cen, bonds
    else:
        raise ValueError(f"unknown molecular preset {preset!r}")

    centres = sorted({tuple(b.centre) for b in basis})
    bonds = [(0, 1)] if len(centres) == 2 else []
    return basis, label, centres, bonds


def huckel_benzene(k):
    """Real Hueckel coefficients for benzene MO k (k = 0 .. 5).

    The complex eigenvectors are c_j = exp(2 pi i j k / 6) / sqrt(6)
    with energies alpha + 2 beta cos(2 pi k / 6); the degenerate pairs
    are combined into the real cosine/sine partners drawn here."""
    k = int(k) % 6
    j = np.arange(6)
    if k == 0:
        return np.full(6, 1.0 / math.sqrt(6.0))
    if k == 5:
        return np.cos(math.pi * j) / math.sqrt(6.0)
    q = (k + 1) // 2              # 1, 1, 2, 2 for k = 1, 2, 3, 4
    ang = 2.0 * math.pi * q * j / 6.0
    if k % 2 == 1:
        return np.cos(ang) / math.sqrt(3.0)
    return np.sin(ang) / math.sqrt(3.0)


def parse_lcao(spec):
    """Parse a custom LCAO specification:

        <orbital>@<x>,<y>,<z>[:zeta] <coefficient> ; ...

    e.g.  "1s@0,0,-1.4 1; 1s@0,0,1.4 -1"  (the sigma* MO of H2).
    Positions are in bohr.  Returns a list of Basis."""
    out = []
    for chunk in str(spec).split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        head, _, coeff = chunk.partition(' ')
        name, at, pos = head.partition('@')
        if not at:
            raise ValueError(f"missing '@position' in LCAO term "
                             f"{chunk!r}")
        pos, _, zeta = pos.partition(':')
        try:
            xyz = [float(v) for v in pos.split(',')]
        except ValueError:
            raise ValueError(f"cannot read the position in LCAO term "
                             f"{chunk!r}")
        if len(xyz) != 3:
            raise ValueError(f"LCAO term {chunk!r} needs three "
                             f"coordinates, got {len(xyz)}")
        try:
            c = float(coeff) if coeff.strip() else 1.0
        except ValueError:
            raise ValueError(f"cannot read the coefficient in LCAO "
                             f"term {chunk!r}")
        out.append(Basis(name, xyz, c,
                         float(zeta) if zeta.strip() else 1.0))
    if not out:
        raise ValueError("the LCAO specification is empty")
    return out


# ==========================================================================
# Mesh helpers
# ==========================================================================

def mesh_components(nverts, tris):
    """Connected-component labels of a triangle soup with welded
    vertices, by union-find over the triangle edges."""
    parent = list(range(nverts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t in tris:
        a = find(int(t[0]))
        for i in (1, 2):
            b = find(int(t[i]))
            if a != b:
                parent[b] = a
    roots = {}
    for t in tris:
        roots.setdefault(find(int(t[0])), 0)
        roots[find(int(t[0]))] += 1
    return roots


def _component_labels(nverts, tris):
    """Union-find component label per triangle."""
    parent = list(range(nverts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t in tris:
        a = find(int(t[0]))
        for i in (1, 2):
            b = find(int(t[i]))
            if a != b:
                parent[b] = a
    return np.array([find(int(t[0])) for t in tris])


def filter_components(verts, tris, min_fraction=0.0, largest_only=False):
    """Drop connected pieces smaller than `min_fraction` of the whole
    (and everything but the biggest if asked).

    Where the level set grazes a sample plane tangentially -- which it
    does at the outer edge of every orbital, since the contour is a
    near-sphere of nearly constant |psi| -- the extractor leaves a
    scatter of grid-scale fragments.  They are discretisation debris,
    not lobes, and would otherwise be reported as such."""
    if not len(tris):
        return verts, tris
    lab = _component_labels(len(verts), tris)
    vals, counts = np.unique(lab, return_counts=True)
    if largest_only:
        keep_labels = {vals[int(np.argmax(counts))]}
    else:
        cut = float(min_fraction) * len(tris)
        keep_labels = {v for v, c in zip(vals, counts) if c >= cut}
        if not keep_labels:
            keep_labels = {vals[int(np.argmax(counts))]}
    keep = tris[np.isin(lab, list(keep_labels))]
    used = np.unique(keep)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[keep]


def center_fit(verts, scale=1.0):
    """Centre on the bounding box and fit the largest extent to a 2 m
    cube (the project-wide convention), then apply `scale`."""
    if not len(verts):
        return verts
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ext = float((hi - lo).max())
    return (verts - 0.5 * (lo + hi)) * (2.0 / ext
                                        if ext > 1e-9 else 1.0) * scale


MOLECULAR_ITEMS = [
    ('SIGMA_1S', "sigma 1s", "H2 bonding sigma from two 1s orbitals"),
    ('SIGMA_STAR_1S', "sigma* 1s", "Antibonding partner, nodal plane "
                                   "between the nuclei"),
    ('SIGMA_2S', "sigma 2s", "Bonding sigma from two 2s orbitals"),
    ('SIGMA_STAR_2S', "sigma* 2s", "Antibonding 2s partner"),
    ('SIGMA_2PZ', "sigma 2pz", "End-on 2pz overlap along the bond"),
    ('SIGMA_STAR_2PZ', "sigma* 2pz", "Antibonding end-on partner"),
    ('PI_2PX', "pi 2px", "Side-on 2px overlap"),
    ('PI_STAR_2PX', "pi* 2px", "Antibonding side-on partner"),
    ('DELTA_3DXY', "delta 3dxy", "Face-on d overlap: a metal-metal "
                                 "delta bond"),
    ('DELTA_STAR_3DXY', "delta* 3dxy", "Antibonding delta partner"),
    ('SP', "sp hybrid", "(2s + 2pz)/sqrt2"),
    ('SP2', "sp2 hybrid", "(2s + sqrt2 2px)/sqrt3"),
    ('SP3', "sp3 hybrid", "(2s + 2px + 2py + 2pz)/2"),
    ('WATER_BOND', "H2O bonding (a1)", "Symmetry-adapted O-H bonding "
                                       "combination -- qualitative"),
    ('WATER_LONE_PAIR', "H2O lone pair (a1)", "The s-p lone pair "
                                              "pointing away from the "
                                              "hydrogens"),
    ('BENZENE_PI', "benzene pi (Hueckel)", "One of the six benzene pi "
                                           "molecular orbitals"),
    ('CUSTOM', "Custom LCAO", "Type the combination in the LCAO field"),
]


def build_orbital(mode='ATOMIC', n=2, l=1, m=0, zeta=1.0,
                  preset='SIGMA_1S', separation=1.4, huckel_k=0,
                  lcao="1s@0,0,-1.4 1; 1s@0,0,1.4 -1",
                  probability=0.90, isolevel=0.0, resolution=96,
                  box=0.0, largest_only=False, despeckle=0.005,
                  shells=1, scale=1.0):
    """Mesh one orbital.  Returns (verts, tris, face_sign, label, info).

    The field handed to marching tetrahedra is level - |psi|, which is
    negative inside the lobes; the extractor winds triangles along the
    field gradient, so that sign convention is what puts the normals
    on the outside.

    With `shells` > 1 the orbital comes out as that many NESTED
    contours -- the probability-cloud picture: each encloses an even
    step of the electron density, and rendering them with decreasing
    opacity outward reads as the density falling off."""
    if mode == 'ATOMIC':
        n, l = int(n), int(l)
        m = int(np.clip(int(m), -l, l))
        if l >= n:
            raise ValueError(f"an orbital needs l < n, got n={n}, "
                             f"l={l}")
        label = orbital_name(n, l, m)
        # built directly from quantum numbers: l > 3 has no
        # conventional chemical name to parse
        basis = [Basis.from_numbers(n, l, m, zeta=zeta, label=label)]
        centres, bonds = [(0.0, 0.0, 0.0)], []
        need = radial_extent(n, l, zeta)
    else:
        if preset == 'CUSTOM':
            basis = parse_lcao(lcao)
            label = "custom LCAO"
            centres = sorted({tuple(b.centre) for b in basis})
            bonds = _auto_bonds(centres)
        else:
            basis, label, centres, bonds = molecular_basis(
                preset, separation, huckel_k)
        need = max(radial_extent(b.n, b.l, b.zeta) for b in basis)
        need += max((float(np.linalg.norm(b.centre)) for b in basis),
                    default=0.0)

    nshell = max(1, int(shells))
    # nested contours enclosing an even spread of the density, the
    # outermost at the requested probability
    probs = [probability * (i + 1) / nshell for i in range(nshell)]
    half = float(box) if box > 0.0 else 1.05 * need
    levels, tight = probability_levels(basis, half, probs)
    if isolevel > 0.0:
        levels, tight = [float(isolevel)], half
        probs = [probability]
    if box > 0.0:
        tight = half
    half = tight
    if not levels or max(levels) <= 0.0:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                np.zeros(0, dtype=int), label, {})

    mst = _toolkit()
    res = int(resolution)
    # every shell is marched over the SAME box and put through ONE
    # centre-and-fit at the end; fitting them individually would scale
    # each to the 2 m cube separately and they would stop nesting
    all_v, all_t, all_sign, all_shell = [], [], [], []
    open_edges, base = 0, 0
    for si, level in enumerate(levels):
        if level <= 0.0:
            continue
        verts, tris = mst.marching_tets(
            lambda X, Y, Z: level - np.abs(evaluate_lcao(basis,
                                                         X, Y, Z)),
            (-half, -half, -half), (half, half, half),
            (res, res, res))
        if not len(tris):
            continue
        open_edges += _boundary_edges(tris)
        verts, tris = filter_components(verts, tris,
                                        min_fraction=despeckle,
                                        largest_only=largest_only)
        if not len(tris):
            continue
        cen = verts[tris].mean(axis=1)
        psi = evaluate_lcao(basis, cen[:, 0], cen[:, 1], cen[:, 2])
        all_sign.append(np.where(psi >= 0.0, 1, -1))
        all_shell.append(np.full(len(tris), si, dtype=int))
        all_v.append(verts)
        all_t.append(tris + base)
        base += len(verts)
    if not all_t:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                np.zeros(0, dtype=int), label, {})

    verts = np.vstack(all_v)
    tris = np.vstack(all_t)
    sign = np.concatenate(all_sign)
    shell = np.concatenate(all_shell)

    # the outermost shell is the one whose lobe count is diagnostic --
    # and it is the LOWEST level, so the prediction has to use that one
    # too or the two would be counting different contours
    outer_level = min(levels)
    outer = tris[shell == shell.max()]
    ncomp = len(mesh_components(len(verts), outer))
    cell = 2.0 * half / max(res, 1)
    if mode == 'ATOMIC':
        want, gap = predicted_surfaces(n, l, m, zeta, outer_level, half)
    else:
        want, gap = None, float('inf')
    info = {'level': outer_level, 'levels': levels,
            'probabilities': probs, 'shells': len(set(shell.tolist())),
            'shell': shell, 'box': half, 'components': ncomp,
            'open_edges': open_edges, 'centres': centres,
            'bonds': bonds, 'basis': basis, 'cell': cell,
            'node_gap': gap, 'predicted': want,
            'resolved': want is None or ncomp == want}
    return center_fit(verts, scale), tris, sign, label, info


def radial_regions(n, l, m, zeta, level, rmax, samples=200000):
    """The intervals of r on which the orbital can reach the contour at
    all, i.e. |R_nl(r)| max|Y_l^m| >= level, plus the width of the
    narrowest gap between them.  Returns (regions, gap).

    The angular factor has to be in the comparison: `level` is a
    contour of |psi| = |R| |Y|, so testing |R| against it directly
    would use a threshold too low by the peak of the harmonic and
    report radial lobes that never reach the surface."""
    ymax = _harmonics().max_abs_harmonic(int(l), int(m))
    r = np.linspace(1e-9, float(rmax), int(samples))
    inside = np.abs(radial(n, l, r, zeta)) * ymax >= level
    if not np.any(inside):
        return [], float('inf')
    edges = np.diff(inside.astype(np.int8))
    starts = list(np.nonzero(edges == 1)[0] + 1)
    ends = list(np.nonzero(edges == -1)[0])
    if inside[0]:
        starts.insert(0, 0)
    if inside[-1]:
        ends.append(len(r) - 1)
    regions = [(float(r[a]), float(r[b])) for a, b in zip(starts, ends)]
    gap = float('inf')
    for (_, hi), (lo, _) in zip(regions[:-1], regions[1:]):
        gap = min(gap, lo - hi)
    return regions, gap


def angular_lobes(l, m):
    """Number of connected regions of the sphere on which the real
    harmonic Y_l^m keeps one sign: l - |m| + 1 latitude bands times
    2|m| meridian sectors (one sector when m = 0)."""
    am = abs(int(m))
    return (int(l) - am + 1) * (2 * am if am else 1)


def predicted_surfaces(n, l, m, zeta, level, rmax):
    """How many closed surfaces the isosurface |psi| = level MUST have,
    derived analytically rather than counted off the mesh.

    A region of {|psi| > level} that reaches the nucleus is a ball and
    contributes one surface; a spherical shell (only possible when
    l = 0, since any l > 0 has angular nodes that cut a shell apart)
    contributes two, an inner sphere and an outer one.  With angular
    nodes present each radial region is cut into `angular_lobes`
    simply-connected blobs, one surface apiece.

    Comparing this with what the extractor actually produced is how
    the operator knows the sample grid was too coarse -- a fused pair
    of lobes shows up as a shortfall, with no heuristic involved."""
    regions, gap = radial_regions(n, l, m, zeta, level, rmax)
    if not regions:
        return 0, gap
    if l == 0:
        # a region starting at the nucleus is a ball, the rest shells
        touches = regions[0][0] <= 4.0 * rmax / 200000.0
        return (1 if touches else 2) + 2 * (len(regions) - 1), gap
    return len(regions) * angular_lobes(l, m), gap


def _auto_bonds(centres):
    """Index pairs of centres separated by no more than 1.3x the
    shortest separation -- enough for the presets we ship."""
    if len(centres) < 2:
        return []
    pts = np.asarray(centres, dtype=float)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(len(pts), 1)
    if not len(iu[0]):
        return []
    dmin = float(np.min(d[iu]))
    return [(int(a), int(b)) for a, b in zip(*iu)
            if d[a, b] <= 1.3 * dmin]


def _boundary_edges(tris):
    """Number of edges with a single incident triangle: a non-zero
    count means the isosurface ran into the sample box."""
    cnt = {}
    for t in tris:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    return sum(1 for v in cnt.values() if v == 1)


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty, StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _PALETTE = {1: (0.85, 0.25, 0.20), -1: (0.20, 0.40, 0.85),
                0: (0.75, 0.75, 0.78)}

    def _material(kind, alpha=1.0, suffix=""):
        base = {1: "Orbital psi +", -1: "Orbital psi -",
                0: "Orbital nuclei"}[kind]
        name = base + suffix
        mat = bpy.data.materials.get(name)
        if mat is None:
            rgb = _PALETTE[kind]
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = (*rgb, alpha)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.45
                if "Alpha" in bsdf.inputs:
                    bsdf.inputs["Alpha"].default_value = alpha
            if alpha < 1.0:
                # EEVEE Next (4.2+) renamed the transparency switch;
                # set whichever this build actually has
                if hasattr(mat, 'surface_render_method'):
                    mat.surface_render_method = 'BLENDED'
                elif hasattr(mat, 'blend_method'):
                    mat.blend_method = 'BLEND'
                if hasattr(mat, 'show_transparent_back'):
                    mat.show_transparent_back = True
                if hasattr(mat, 'use_backface_culling'):
                    mat.use_backface_culling = False
        return mat

    def _shell_alpha(si, nshell, base_alpha):
        """Innermost shell nearly opaque, outermost at `base_alpha`."""
        if nshell <= 1:
            return 1.0
        t = si / float(nshell - 1)          # 0 inner .. 1 outer
        return (1.0 - t) * min(1.0, base_alpha + 0.55) + t * base_alpha

    def _new_object(context, name, verts, faces, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth',
                                [smooth] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    class MESH_OT_orbital_add(bpy.types.Operator):
        """Add a hydrogenic atomic orbital or an LCAO molecular orbital
        as a sign-coloured probability isosurface"""
        bl_idname = "mesh.orbital_add"
        bl_label = "Atomic & Molecular Orbital"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('ATOMIC', "Atomic", "One hydrogenic orbital "
                                        "psi_nlm"),
                   ('MOLECULAR', "Molecular", "A linear combination of "
                                              "atomic orbitals")],
            default='ATOMIC')
        n: IntProperty(
            name="n", default=2, min=1, max=6,
            description="Principal quantum number: the orbital has "
                        "n-l-1 radial nodes")
        l: IntProperty(
            name="l", default=1, min=0, max=5,
            description="Azimuthal quantum number (0 = s, 1 = p, "
                        "2 = d, 3 = f); must be < n")
        m: IntProperty(
            name="m", default=0, min=-5, max=5,
            description="Magnetic quantum number of the real orbital; "
                        "clamped to |m| <= l")
        zeta: FloatProperty(
            name="Zeta", default=1.0, min=0.1, max=10.0,
            description="Effective nuclear charge: 1 is hydrogen, "
                        "larger values contract the orbital "
                        "(a Slater-type basis function)")
        preset: EnumProperty(
            name="Molecule", items=MOLECULAR_ITEMS,
            default='SIGMA_1S')
        separation: FloatProperty(
            name="Bond Length", default=1.4, min=0.2, max=12.0,
            description="Diatomic nuclear separation in bohr "
                        "(1.4 a0 is H2)")
        huckel_k: IntProperty(
            name="Hueckel MO", default=0, min=0, max=5,
            description="Which of the six benzene pi orbitals to draw, "
                        "lowest energy first")
        lcao: StringProperty(
            name="LCAO",
            default="1s@0,0,-1.4 1; 1s@0,0,1.4 -1",
            description="Custom combination: "
                        "orbital@x,y,z[:zeta] coefficient, "
                        "semicolon separated (positions in bohr)")
        probability: FloatProperty(
            name="Enclosed Probability", default=0.90, min=0.05,
            max=0.99,
            description="Fraction of the electron density the "
                        "isosurface encloses")
        isolevel: FloatProperty(
            name="Level Override", default=0.0, min=0.0, max=10.0,
            description="Raw |psi| contour value; 0 uses the enclosed "
                        "probability instead")
        resolution: IntProperty(
            name="Resolution", default=96, min=24, max=192,
            description="Sample grid resolution per axis (cost grows "
                        "as the cube)")
        box: FloatProperty(
            name="Box Override", default=0.0, min=0.0, max=200.0,
            description="Sample box half-width in bohr; 0 fits it to "
                        "the orbital")
        skeleton: BoolProperty(
            name="Nuclei & Bonds", default=True,
            description="Add a ball-and-stick skeleton for the nuclei "
                        "(molecular mode)")
        sign_colors: BoolProperty(
            name="Sign Colours", default=True,
            description="Two material slots for the positive and "
                        "negative lobes of psi")
        largest_only: BoolProperty(
            name="Largest Lobe Only", default=False,
            description="Discard all but the biggest connected piece")
        display: EnumProperty(
            name="Display",
            items=[('SOLID', "Single Surface", "One isosurface at the "
                                               "enclosed probability"),
                   ('CLOUD', "Probability Cloud", "Nested transparent "
                                                  "contours, fading "
                                                  "outward as the "
                                                  "density falls off")],
            default='SOLID')
        shells: IntProperty(
            name="Cloud Shells", default=3, min=2, max=6,
            description="How many nested contours the cloud has; each "
                        "encloses an even step of the electron density")
        cloud_alpha: FloatProperty(
            name="Outer Opacity", default=0.18, min=0.02, max=1.0,
            description="Opacity of the outermost shell; the inner "
                        "ones are progressively more solid")
        despeckle: FloatProperty(
            name="Despeckle", default=0.005, min=0.0, max=0.2,
            description="Drop connected pieces smaller than this "
                        "fraction of the mesh: grid-scale debris where "
                        "the contour grazes a sample plane")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this "
                        "thickness")
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            if self.mode == 'ATOMIC' and self.l >= self.n:
                self.report({'ERROR'},
                            f"l must be less than n (got n={self.n}, "
                            f"l={self.l})")
                return {'CANCELLED'}
            if self.mode == 'ATOMIC' and abs(self.m) > self.l:
                self.report({'WARNING'},
                            f"|m| must be <= l; clamping m to "
                            f"{int(np.clip(self.m, -self.l, self.l))}")
            try:
                verts, tris, sign, label, info = build_orbital(
                    mode=self.mode, n=self.n, l=self.l, m=self.m,
                    zeta=self.zeta, preset=self.preset,
                    separation=self.separation, huckel_k=self.huckel_k,
                    lcao=self.lcao, probability=self.probability,
                    isolevel=self.isolevel, resolution=self.resolution,
                    box=self.box, largest_only=self.largest_only,
                    despeckle=self.despeckle, scale=self.scale,
                    shells=(self.shells
                            if self.display == 'CLOUD' else 1))
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            if not len(tris):
                self.report({'ERROR'},
                            "Empty isosurface -- try a lower enclosed "
                            "probability")
                return {'CANCELLED'}

            faces = [tuple(int(i) for i in t) for t in tris]
            nsurf = len(faces)
            if (self.mode == 'MOLECULAR' and self.skeleton
                    and info.get('centres')):
                faces = self._append_skeleton(verts, faces, info)
                verts, faces = self._verts, faces

            obj = _new_object(context, label, verts, faces,
                              smooth=self.smooth)
            me = obj.data
            if len(me.polygons) == len(faces):
                nsh = int(info.get('shells', 1))
                shell = info.get('shell')
                if self.display == 'CLOUD' and nsh > 1 and shell is not None:
                    # one slot per (shell, sign): the innermost shell
                    # is nearly solid, the outermost a faint haze
                    slot, idx = {}, []
                    for si in range(nsh):
                        a = _shell_alpha(si, nsh, self.cloud_alpha)
                        for s in ((1, -1) if self.sign_colors else (1,)):
                            slot[(si, s)] = len(me.materials)
                            me.materials.append(
                                _material(s, alpha=a,
                                          suffix=f" cloud {si + 1}"
                                                 f"/{nsh}"))
                    nuc = len(me.materials)
                    me.materials.append(_material(0))
                    for k in range(nsurf):
                        s = int(sign[k]) if self.sign_colors else 1
                        idx.append(slot[(int(shell[k]), s)])
                    idx += [nuc] * (len(faces) - nsurf)
                    me.polygons.foreach_set('material_index', idx)
                    me.update()
                elif self.sign_colors:
                    me.materials.append(_material(1))
                    me.materials.append(_material(-1))
                    me.materials.append(_material(0))
                    idx = [0 if s > 0 else 1 for s in sign]
                    idx += [2] * (len(faces) - nsurf)
                    me.polygons.foreach_set('material_index', idx)
                    me.update()
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            if info.get('open_edges'):
                self.report({'WARNING'},
                            f"{label}: the isosurface reached the "
                            f"sample box; raise Box Override or lower "
                            f"the enclosed probability")
            if not info.get('resolved', True):
                gap = info['node_gap']
                why = (f"the narrowest radial node gap is "
                       f"{gap:.2f} a0 against a "
                       f"{info['cell']:.2f} a0 sample cell"
                       if math.isfinite(gap) else
                       f"a {info['cell']:.2f} a0 sample cell cannot "
                       f"separate lobes across the angular nodes")
                self.report({'WARNING'},
                            f"{label}: drew {info['components']} of "
                            f"the {info['predicted']} lobe surfaces "
                            f"this orbital has -- {why}; raise "
                            f"Resolution")
            if self.display == 'CLOUD':
                pcts = ", ".join(f"{100.0 * p:.0f}%"
                                 for p in info.get('probabilities', ()))
                self.report({'INFO'},
                            f"{label}: {len(me.vertices)} verts, "
                            f"{len(me.polygons)} faces, "
                            f"{info.get('shells', 1)} shells enclosing "
                            f"{pcts} of the density")
            else:
                self.report({'INFO'},
                            f"{label}: {len(me.vertices)} verts, "
                            f"{len(me.polygons)} faces, "
                            f"{info.get('components', 0)} lobes, "
                            f"|psi| = {info.get('level', 0.0):.4g}")
            return {'FINISHED'}

        def _append_skeleton(self, verts, faces, info):
            """Append a ball-and-stick model of the nuclei, in the same
            coordinate frame the surface was fitted into."""
            try:
                from . import ball_and_stick as bas
            except ImportError:
                import ball_and_stick as bas
            half = info['box']
            # the surface was centred and fitted to a 2 m cube; put the
            # nuclei through the same map so the two line up
            src = np.asarray(info['centres'], dtype=float)
            k = self.scale / max(half, 1e-9)
            pts = [tuple(p) for p in src * k]
            bonds = info['bonds'] or _auto_bonds(info['centres'])
            self._verts = verts
            if not bonds:
                # a one-centre preset (the sp/sp2/sp3 hybrids, a lone
                # pair): ball-and-stick emits nodes only for vertices an
                # edge uses, so there is nothing to append -- and the
                # single nucleus sits buried inside the lobe anyway
                return faces
            sv, sf = bas.build_mesh(pts, bonds,
                                    strut_radius=0.02 * self.scale,
                                    node_radius=0.05 * self.scale)
            if not sv or not sf:
                return faces
            off = len(verts)
            self._verts = np.vstack([verts,
                                     np.asarray(sv, dtype=float)])
            return faces + [tuple(off + i for i in f) for f in sf]

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'ATOMIC':
                row = lay.row(align=True)
                row.prop(self, 'n')
                row.prop(self, 'l')
                row.prop(self, 'm')
                lay.prop(self, 'zeta')
            else:
                lay.prop(self, 'preset')
                if self.preset == 'CUSTOM':
                    lay.prop(self, 'lcao')
                elif self.preset == 'BENZENE_PI':
                    lay.prop(self, 'huckel_k')
                elif self.preset.startswith(('SIGMA', 'PI', 'DELTA')):
                    lay.prop(self, 'separation')
                lay.prop(self, 'skeleton')
            lay.prop(self, 'display')
            if self.display == 'CLOUD':
                lay.prop(self, 'shells')
                lay.prop(self, 'cloud_alpha')
            lay.prop(self, 'probability')
            lay.prop(self, 'isolevel')
            for k in ('resolution', 'box', 'sign_colors', 'despeckle',
                      'largest_only', 'scale', 'thickness', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.orbital_add", icon='META_BALL')

    _classes = (MESH_OT_orbital_add,)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


# ==========================================================================
# Standalone numeric self-test
# ==========================================================================

def _selftest():
    # ---- 1. radial normalisation and orthogonality ------------------
    r = np.linspace(1e-8, 260.0, 400000)
    dr = r[1] - r[0]
    worst = 0.0
    for n in range(1, 6):
        for l in range(n):
            R = radial(n, l, r)
            val = float(np.sum(R * R * r * r) * dr)
            worst = max(worst, abs(val - 1.0))
            for n2 in range(l + 1, 6):
                if n2 == n:
                    continue
                R2 = radial(n2, l, r)
                ov = abs(float(np.sum(R * R2 * r * r) * dr))
                worst = max(worst, ov)
    if worst > 1e-4:
        raise AssertionError(
            f"radial functions are not orthonormal: worst deviation "
            f"{worst:.3e}")
    print(f"radial orthonormality (n <= 5): worst deviation {worst:.2e}")

    # ---- 2. radial node counts (n - l - 1) --------------------------
    # this is the assertion that catches the classic wrong Laguerre
    # index L_(n+l)^(2l+1) instead of L_(n-l-1)^(2l+1)
    rr = np.linspace(1e-6, 200.0, 200001)
    for n in range(1, 7):
        for l in range(n):
            R = radial(n, l, rr)
            R = R[np.abs(R) > 1e-30]          # ignore the dead tail
            s = np.sign(R)
            nz = int(np.sum(s[1:] * s[:-1] < 0))
            if nz != n - l - 1:
                raise AssertionError(
                    f"R({n},{l}) has {nz} radial nodes, expected "
                    f"{n - l - 1}")
    print("radial node counts (n <= 6): n-l-1 nodes throughout")

    # ---- 3. closed forms --------------------------------------------
    t = np.linspace(0.0, 30.0, 3001)
    checks = {
        (1, 0): 2.0 * np.exp(-t),
        (2, 0): (2.0 - t) * np.exp(-0.5 * t) / (2.0 * math.sqrt(2.0)),
        (2, 1): t * np.exp(-0.5 * t) / (2.0 * math.sqrt(6.0)),
    }
    for (n, l), ref in checks.items():
        err = float(np.max(np.abs(radial(n, l, t) - ref)))
        if err > 1e-12:
            raise AssertionError(
                f"R({n},{l}) differs from its closed form by {err:.3e}")
    print("closed forms R10, R20, R21 reproduced to 1e-12")

    # ---- 4. orbital name parsing ------------------------------------
    for name, want in (('1s', (1, 0, 0)), ('2pz', (2, 1, 0)),
                       ('2px', (2, 1, 1)), ('2py', (2, 1, -1)),
                       ('3dxy', (3, 2, -2)), ('3dz2', (3, 2, 0)),
                       ('4fz3', (4, 3, 0))):
        got = parse_orbital(name)
        if got != want:
            raise AssertionError(f"parse_orbital({name!r}) gave {got}, "
                                 f"expected {want}")
    for bad in ('1p', '2q', 'xx', '2p', '0s'):
        try:
            parse_orbital(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_orbital({bad!r}) should have "
                             f"raised")

    # ---- 5. LCAO parser round trip ----------------------------------
    bs = parse_lcao("1s@0,0,-1.4 1; 2pz@0,0,1.4:1.5 -0.5")
    if len(bs) != 2:
        raise AssertionError(f"LCAO parser produced {len(bs)} terms, "
                             f"expected 2")
    if (bs[0].n, bs[0].l, bs[0].m) != (1, 0, 0) or bs[0].coeff != 1.0:
        raise AssertionError("first LCAO term parsed wrongly")
    if (bs[1].n, bs[1].l, bs[1].m) != (2, 1, 0) or bs[1].zeta != 1.5:
        raise AssertionError("second LCAO term parsed wrongly")
    for bad in ("1s 1", "1s@0,0 1", "1s@a,b,c 1", ""):
        try:
            parse_lcao(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_lcao({bad!r}) should have raised")
    print("orbital and LCAO parsers accept the good and reject the bad")

    # ---- 6. Hueckel benzene -----------------------------------------
    C = np.array([huckel_benzene(k) for k in range(6)])
    gram = C @ C.T
    if float(np.max(np.abs(gram - np.eye(6)))) > 1e-12:
        raise AssertionError("benzene Hueckel vectors are not "
                             "orthonormal")
    # adjacency eigenvalues: 2, 1, 1, -1, -1, -2 (energies alpha + x beta)
    A = np.zeros((6, 6))
    for j in range(6):
        A[j, (j + 1) % 6] = A[(j + 1) % 6, j] = 1.0
    for k, want in enumerate((2.0, 1.0, 1.0, -1.0, -1.0, -2.0)):
        got = float(C[k] @ A @ C[k])
        if abs(got - want) > 1e-12:
            raise AssertionError(
                f"benzene MO {k} has energy alpha {got:+.4f} beta, "
                f"expected {want:+.1f}")
    print("benzene Hueckel MOs orthonormal with energies "
          "2, 1, 1, -1, -1, -2 beta")

    # ---- 7. atomic isosurfaces --------------------------------------
    # Counting SURFACES, not solid lobes: a ball contributes one closed
    # surface, a shell two (an inner and an outer sphere).  So 1s is 1,
    # 2s is 1 + 2 = 3, 2pz is two lobes = 2, 3pz is two lobes each cut
    # in two by its radial node = 4, and 3dxy / 4fz3 / 5g_z4 are 4, 4
    # and 5 purely angular lobes (l - |m| latitude nodes leave
    # l - |m| + 1 of them).
    want_lobes = {(1, 0, 0): 1, (2, 0, 0): 3, (2, 1, 0): 2,
                  (3, 1, 0): 4, (3, 2, -2): 4, (4, 3, 0): 4,
                  (5, 4, 0): 5}
    for (n, l, m), want in want_lobes.items():
        V, T, S, label, info = build_orbital(
            mode='ATOMIC', n=n, l=l, m=m, resolution=96,
            probability=0.90)
        if info['predicted'] != want:
            raise AssertionError(
                f"{label}: the analytic surface count says "
                f"{info['predicted']}, the mathematics says {want}")
        if not len(T):
            raise AssertionError(f"{label} produced an empty surface")
        if info['open_edges']:
            raise AssertionError(
                f"{label} ran into the sample box "
                f"({info['open_edges']} open edges)")
        if info['components'] != want:
            raise AssertionError(
                f"{label} has {info['components']} lobes, expected "
                f"{want}")
        if not np.all(np.isfinite(V)):
            raise AssertionError(f"{label} produced non-finite verts")
        ext = float((V.max(axis=0) - V.min(axis=0)).max())
        if abs(ext - 2.0) > 1e-6:
            raise AssertionError(f"{label} is {ext:.4f} across, "
                                 f"expected a 2 m fit")
        pos, neg = int(np.sum(S > 0)), int(np.sum(S < 0))
        print(f"{label:5s}: {len(V):6d} verts {len(T):6d} tris "
              f"{info['components']} lobes  "
              f"{pos}+/{neg}- faces  level {info['level']:.4g}")
        if l > 0 and min(pos, neg) == 0:
            raise AssertionError(f"{label} should have both signs")

    # an under-resolved node must be REPORTED, not hidden: 3s has five
    # lobe surfaces (a ball inside two shells) but its innermost node
    # gap is 0.24 a0 against a 0.44 a0 cell at the default resolution,
    # so the extractor fuses lobes and the operator has to say so
    V, T, S, label, info = build_orbital(mode='ATOMIC', n=3, l=0, m=0,
                                         resolution=96,
                                         probability=0.90)
    if info['predicted'] != 5:
        raise AssertionError(f"3s should have 5 lobe surfaces, the "
                             f"analytic count says {info['predicted']}")
    if info['resolved'] or info['components'] >= info['predicted']:
        raise AssertionError(
            f"3s at resolution 96 should come up short and be flagged "
            f"(drew {info['components']} of {info['predicted']})")
    print(f"3s   : correctly flagged -- drew {info['components']} of "
          f"{info['predicted']} surfaces (gap {info['node_gap']:.2f} "
          f"a0 vs cell {info['cell']:.2f} a0)")

    # ---- 8. molecular orbitals --------------------------------------
    # the antibonding partner has a nodal plane between the nuclei, so
    # it comes apart into two pieces where the bonding one is single
    for preset, want in (('SIGMA_1S', 1), ('SIGMA_STAR_1S', 2),
                         ('PI_2PX', 2), ('PI_STAR_2PX', 4)):
        V, T, S, label, info = build_orbital(
            mode='MOLECULAR', preset=preset, resolution=64,
            probability=0.90)
        if info['components'] != want:
            raise AssertionError(
                f"{label} has {info['components']} lobes, expected "
                f"{want}")
        print(f"{label:28s}: {info['components']} lobes, "
              f"{len(T):6d} tris")

    # psi must vanish on the plane between the nuclei of sigma*
    basis, _, _, _ = molecular_basis('SIGMA_STAR_1S', 1.4)
    mid = evaluate_lcao(basis, np.array([0.0, 0.3, -0.3]),
                        np.array([0.0, -0.2, 0.4]),
                        np.zeros(3))
    if float(np.max(np.abs(mid))) > 1e-12:
        raise AssertionError("sigma* 1s does not vanish on its nodal "
                             "plane")

    for preset in ('SP', 'SP2', 'SP3', 'WATER_BOND',
                   'WATER_LONE_PAIR', 'BENZENE_PI'):
        V, T, S, label, info = build_orbital(
            mode='MOLECULAR', preset=preset, resolution=56,
            probability=0.85)
        if not len(T):
            raise AssertionError(f"{label} produced an empty surface")
        if not np.all(np.isfinite(V)):
            raise AssertionError(f"{label} produced non-finite verts")
        print(f"{label:40s}: {info['components']} lobes, "
              f"{len(T):6d} tris")

    # ---- 9. probability cloud: the shells must actually nest --------
    # each contour encloses more density than the one inside it, so its
    # level is LOWER and its extent strictly LARGER.  All shells share
    # one centre-and-fit, which is what keeps them concentric -- fitting
    # each to the 2 m cube on its own would stack them exactly.
    for kw in (dict(mode='ATOMIC', n=2, l=1, m=0),
               dict(mode='MOLECULAR', preset='SIGMA_1S'),
               dict(mode='MOLECULAR', preset='PI_STAR_2PX')):
        V, T, S, label, info = build_orbital(resolution=56, shells=3,
                                             probability=0.90, **kw)
        if info['shells'] != 3:
            raise AssertionError(f"{label} cloud has {info['shells']} "
                                 f"shells, expected 3")
        lv = info['levels']
        if not all(a > b for a, b in zip(lv[:-1], lv[1:])):
            raise AssertionError(
                f"{label} cloud levels {lv} are not decreasing outward")
        shell = info['shell']
        prev = -1.0
        for si in range(info['shells']):
            sv = V[np.unique(T[shell == si])]
            reach = float(np.max(np.linalg.norm(sv, axis=1)))
            if reach <= prev:
                raise AssertionError(
                    f"{label} cloud shell {si} reaches {reach:.3f}, "
                    f"not beyond the {prev:.3f} of the one inside it")
            prev = reach
        ext = float((V.max(axis=0) - V.min(axis=0)).max())
        if abs(ext - 2.0) > 1e-6:
            raise AssertionError(f"{label} cloud is {ext:.4f} across, "
                                 f"expected a 2 m fit")
        print(f"{label:28s}: cloud of {info['shells']} nested shells, "
              f"levels {['%.4g' % v for v in lv]}, {len(T)} tris")

    # custom LCAO reproduces the sigma* preset
    V, T, S, label, info = build_orbital(
        mode='MOLECULAR', preset='CUSTOM',
        lcao="1s@0,0,-0.7 1; 1s@0,0,0.7 -1", resolution=56)
    if info['components'] != 2:
        raise AssertionError(
            f"custom sigma* has {info['components']} lobes, expected 2")
    print(f"custom LCAO sigma*: {info['components']} lobes, "
          f"{len(T):6d} tris")

    print("RESULT: OK")
