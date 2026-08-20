# Atomic orbital isosurfaces.
#
# Part of the Math Art minsurf engine (`math_art/minsurf/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# The hydrogen-like wavefunctions psi_nlm, contoured at a fixed
# probability density.  The angular part is a spherical harmonic and
# the radial part an associated Laguerre polynomial.

import math
import re
import numpy as np


def _toolkit():
    """`marching_tets` lives in this package's own `tpms` module.

    This module IS inside `minsurf`, so it reaches a sibling with one
    dot -- it does not import the package it lives in.
    """
    from . import tpms as mst
    return mst


def _harmonics():
    """The sibling Spherical Harmonic module supplies real_sph_harm."""
    # the spherical-harmonic engine moved into `surfaces` in the same
    # pass that put this module here
    try:
        from ..surfaces import harmonics as shg
    except ImportError:
        from surfaces import harmonics as shg
    return shg


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
    # |psi| is the same field for EVERY shell -- only the level moves --
    # so sample it once and hand the extractor the grid.  Marching each
    # shell from a callable re-ran the whole LCAO evaluation per shell,
    # which is by far the most expensive part of this build.
    ax = np.linspace(-half, half, res + 1)
    Xg, Yg, Zg = np.meshgrid(ax, ax, ax, indexing='ij')
    absmap = np.abs(evaluate_lcao(basis, Xg, Yg, Zg))
    del Xg, Yg, Zg
    for si, level in enumerate(levels):
        if level <= 0.0:
            continue
        verts, tris = mst.marching_tets(
            level - absmap,
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
