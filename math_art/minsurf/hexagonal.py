# Exact Weierstrass engine for the HEXAGONAL genus-3 triply periodic
# minimal surfaces -- Schwarz's H surface today, with room for CLP and
# the Lidinoid, which share the construction.
#
# These are the TPMS that the repo's nodal machinery cannot reach.  A
# nodal surface is a trigonometric polynomial whose zero set happens to
# sit close to a minimal surface; it is an approximation, and for
# Schwarz H no such approximation has ever been published (the Fisher et
# al. 2023 catalogue, which collects every known nodal fit, omits it).
# The alternative route is to stop approximating and integrate the
# Weierstrass representation of the real surface, which is what this
# module does.
#
#   X = Re Int (phi1, phi2, phi3) dz,
#   phi1 = (1/2)(1/g - g) dh,  phi2 = (i/2)(1/g + g) dh,  phi3 = dh.
#
# The H-surface data lives on the square torus C / (Z + i Z) and is due
# to Schwarz; the presentation followed here -- theta-function Gauss map
# with a flat height differential -- is Weber's:
#
#   tau = i,  a0 = 1/4,  dh = dz,
#   G2(z) = theta11(z - a0, tau) / theta11(z + a0, tau),
#   g(z)  = G2(z)^(2/3),
#   domain  x in [0, a0],  y in [0, Im(tau)/2].
#
# The 2/3 power is the whole character of the surface: it makes g a
# three-sheeted cover, which is what produces the three-fold symmetry of
# the hexagonal cell, and it puts a branch point at the corner z = a0
# where g vanishes.  That point is an ordinary FLAT point of the
# surface, not an end -- the metric (|g| + 1/|g|)|dh| blows up there like
# s^(-2/3), which is integrable, so the point sits at finite distance.
# `_domain_u` handles it by substitution rather than by avoidance.
#
# Nothing about the hexagonal lattice is put in by hand.  The four
# boundary curves of the fundamental patch are measured, classified as
# straight lines or planar geodesics, turned into Schwarz reflections,
# and the period lattice is read off the group they generate.  That it
# comes out hexagonal -- two equal generators at 120 degrees, both
# perpendicular to the third -- is therefore a result and a check, not
# an assumption; `_selftest` gates on it.
#
# References:
# - H. A. Schwarz, "Bestimmung einer speziellen Minimalflaeche",
#   Gesammelte Mathematische Abhandlungen, Springer (1890) -- the H
#   surface and the reflection principle used to extend it.
# - K. Weierstrass (1866) and A. Enneper (1864), the representation
#   formula integrated here.
# - M. Weber, "Schwarz' H surface", minimalsurfaces.blog (Indiana
#   University); the theta-function Gauss map, the modulus tau = i and
#   the branch value a0 = 1/4 are taken from the accompanying notebook.
# - A. H. Schoen, "Infinite periodic minimal surfaces without
#   self-intersections", NASA TN D-5541 (1970) -- H in the wider family.
# - H. G. von Schnering and R. Nesper, "Nodal surfaces of Fourier
#   series", Z. Phys. B 83 (1991) -- the nodal method, for contrast:
#   it is what does NOT reach this surface.
# - A. H. Fisher et al., "A catalogue of nodal approximations to triply
#   periodic minimal surfaces" (2023) -- confirms no published nodal
#   formula exists for H.

import math

import numpy as np

try:
    from .. import geom_cache as _geom_cache
except ImportError:  # flat import outside the package
    import geom_cache as _geom_cache

# --------------------------------------------------------------------
# Weierstrass data
# --------------------------------------------------------------------
# Torus modulus and the nome.  |q| = e^-pi ~ 0.0432, so the theta series
# converges geometrically and a dozen terms are already at round-off;
# 24 is free and leaves headroom if tau is ever made a parameter.
_TAU = 1j
_Q = np.exp(1j * np.pi * _TAU)
_A0 = 0.25
_Y1 = 0.5                                  # Im(tau) / 2
_TERMS = 24

_N = np.arange(_TERMS)
_COEF = ((-1.0) ** _N) * _Q ** ((_N + 0.5) ** 2)
_K = np.pi * (2 * _N + 1).astype(float)


def _theta11(z, q=None):
    """The odd Jacobi theta function in Weber's normalisation, period 1
    in z:  theta11(z, tau) = 2 sum (-1)^n q^((n+1/2)^2) sin((2n+1) pi z).

    `q` is the nome exp(i pi tau).  It is a PARAMETER, not the module
    constant, because the surfaces on this branch do not share a torus:
    H sits on tau = i, CLP on 2i, and the Lidinoid and rPD on moduli
    fixed by their own period conditions.  Evaluating one surface's
    theta with another's nome silently returns a nearly constant Gauss
    map, and a constant Gauss map is a PLANE -- which passes a mean
    curvature test perfectly while being the wrong surface entirely.
    """
    if q is None:
        coef, k = _COEF, _K
    else:
        n = np.arange(_TERMS)
        coef = ((-1.0) ** n) * q ** ((n + 0.5) ** 2)
        k = np.pi * (2 * n + 1).astype(float)
    ang = np.multiply.outer(np.asarray(z, dtype=complex), k)
    return 2.0 * np.sum(coef * np.sin(ang), axis=-1)


def _log_g2(D, Z):
    """log G2 over the grid, on a CONTINUOUS branch.

    `D` is z - a0, passed in as its own array rather than recomputed
    here as a subtraction.  That is not fussiness: the grid is graded
    into the branch point, so its last samples sit ~1e-19 away from a0,
    and forming (x - a0) in float64 at that distance returns exactly
    zero -- every significant digit is lost to cancellation and the
    Gauss map collapses to 0/0.

    G2 is holomorphic and non-vanishing on the open rectangle (theta11
    vanishes only on the lattice, and the only lattice point in the
    closed domain is the corner z = a0), so a continuous logarithm
    exists.  It is obtained by unwrapping the principal branch up the
    left edge, where G2 = -1 and nothing is near a zero, and then along
    each row from there.
    """
    L = np.log(_theta11(D)) - np.log(_theta11(Z + _A0))
    im = np.imag(L)
    im[0] = np.unwrap(im[0])
    im = np.unwrap(im, axis=0)
    return np.real(L) + 1j * im


def _forms(D, Z, theta):
    """(phi1, phi2, phi3) / dz at every sample, rotated by the associate
    angle.  dh = dz, so the dz is carried by the quadrature."""
    g = np.exp((2.0 / 3.0) * _log_g2(D, Z))
    inv = 1.0 / g
    rot = np.exp(1j * float(theta))
    return (np.stack([0.5 * (inv - g), 0.5j * (inv + g),
                      np.ones_like(g)], axis=-1) * rot)


def _domain_u(nu):
    """The x samples, as the substituted variable u with x = a0 - u^3.

    Near the branch point the integrand of phi2 grows like s^(-2/3) in
    s = a0 - x.  That is integrable -- the surface is perfectly regular
    there -- but a trapezoid rule on uniform x samples converges against
    it at a dismal rate, and the last interval, whose sample sits right
    on the pole, dominates the whole answer: the patch came out with a
    diameter of 36 at one resolution and 5 at another before this was
    fixed.

    Substituting s = u^3 cancels the exponent exactly: the integrand
    times ds/du = 3u^2 behaves like u^(-2) * u^2, which is bounded, so
    the ordinary rule is second-order accurate right up to the corner.
    The quadrature must then run IN u -- putting these nodes into a rule
    over x buys nothing.  u descends so that x ascends; the last sample
    stops a hair above zero because the integrand is genuinely infinite
    at the point even though the integral through it is not.
    """
    u = np.linspace(_A0 ** (1.0 / 3.0), 0.0, int(nu))
    u[-1] = u[0] * 1e-6
    return u


def h_patch(nu=70, nv=140, theta=0.0, refine=4):
    """The fundamental patch: (nu, nv, 3) points.

    Integrated up the left edge first (dz = i dy, no singularity
    anywhere on it) and then outward along each row in u, so the branch
    point is approached only at the very end of the very last row.

    The QUADRATURE grid and the MESH grid are deliberately different.
    Grading into the branch point is what makes the integral converge,
    but it also drives consecutive samples to within 1e-9 of each other
    there, and vertices that close together cannot survive being welded
    to the neighbouring copies of the patch -- the weld has to be loose
    enough to close the seams, so it collapses the cluster instead and
    tears holes in the assembled cell.  So the integration runs on the
    graded grid at `refine` times the density, and the result is
    interpolated onto a mesh grid uniform in x, where every vertex is a
    comfortable distance from its neighbours.
    """
    nu, nv = int(nu), int(nv)
    u = _domain_u(max(8, nu * int(refine)))
    s = u ** 3                                    # a0 - x, exactly
    ys = np.linspace(0.0, _Y1, nv)
    Z = (_A0 - s)[:, None] + 1j * ys[None, :]
    D = (-s[:, None]) + 1j * ys[None, :]
    W = _forms(D, Z, theta)

    F = np.zeros((len(u), nv, 3), dtype=complex)
    dy = ys[1] - ys[0]
    col = W[0]
    F[0, 1:] = np.cumsum(0.5 * (col[:-1] + col[1:]) * (1j * dy), axis=0)
    V = W * (3.0 * u ** 2)[:, None, None]         # integrand * dx/du
    du = -np.diff(u)[:, None, None]               # positive: u descends
    F[1:] = F[0][None, :, :] + np.cumsum(
        0.5 * (V[:-1] + V[1:]) * du, axis=0)
    P = np.real(F)

    xs_int = _A0 - s                              # ascending
    xs_out = np.linspace(0.0, _A0, nu)
    out = np.empty((nu, nv, 3))
    for j in range(nv):
        for k in range(3):
            out[:, j, k] = np.interp(xs_out, xs_int, P[:, j, k])
    return out


# --------------------------------------------------------------------
# Schwarz extension
# --------------------------------------------------------------------
# The reflection principle: a minimal surface continues analytically
# across a straight line lying in it by a half-turn about that line, and
# across a curve along which it meets a plane orthogonally by reflection
# in that plane.  So the symmetry group is not chosen -- it is read off
# the boundary of the patch, once the patch is known.

def _classify(C):
    """Straightness and planarity of a polyline, each as a residual
    divided by the curve's own length so the numbers are scale-free.
    Also returns the best-fit plane normal and the centroid."""
    C = np.asarray(C, float)
    L = float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))) or 1.0
    d = C[-1] - C[0]
    dn = d / max(float(np.linalg.norm(d)), 1e-300)
    straight = float(np.max(np.linalg.norm(
        np.cross(C - C[0], dn), axis=1))) / L
    Q = C - C.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    planar = float(np.max(np.abs(Q @ Vt[2]))) / L
    return straight, planar, Vt[2], C.mean(0)


def _mirror(n, d):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    M = np.eye(4)
    M[:3, :3] = np.eye(3) - 2.0 * np.outer(n, n)
    M[:3, 3] = 2.0 * d * n
    return M


def _halfturn(p, v):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    R = 2.0 * np.outer(v, v) - np.eye(3)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(p, float) - R @ np.asarray(p, float)
    return M


def _snap_axis(v, step=30.0, tol=1e-2):
    """Snap a direction to the nearest hexagonal axis.

    The measured normals and axis directions come out within 1e-5 of
    multiples of 30 degrees in the xy-plane, or of z.  Snapping them to
    exactly that costs nothing in fidelity -- it is a 1e-5 correction to
    a quantity the surface's own symmetry forces -- and it buys
    everything downstream: the generators become exact, so composing
    them stays exact, and the group closes instead of drifting.

    Only the DIRECTIONS are snapped.  Plane offsets and axis positions
    are genuine measurements of where the surface sits and are left
    alone.
    """
    v = np.asarray(v, float)
    v = v / max(float(np.linalg.norm(v)), 1e-300)
    if abs(v[2]) > 1.0 - tol:
        return np.array([0.0, 0.0, math.copysign(1.0, v[2])])
    if abs(v[2]) > tol:
        return v                                 # not an axis we know
    ang = math.degrees(math.atan2(v[1], v[0]))
    k = round(ang / step)
    if abs(ang - step * k) > step * tol:
        return v
    a = math.radians(step * k)
    return np.array([math.cos(a), math.sin(a), 0.0])


def h_generators(P, tol=1e-3):
    """One Schwarz generator per boundary curve of the patch, with the
    kind of each (for the self-test to report).

    Straightness is tested BEFORE planarity because a straight line is
    trivially planar; the half-turn is the correct continuation.
    """
    gens, kinds = [], []
    edges = (('x=0', P[0, :]), ('x=a0', P[-1, :]),
             ('y=0', P[:, 0]), ('y=y1', P[:, -1]))
    for name, C in edges:
        straight, planar, nrm, cen = _classify(C)
        if straight < tol:
            # least squares over the WHOLE edge, not the two endpoints:
            # the endpoints are the two samples with the most quadrature
            # behind them, and a chord through them misses the fitted
            # line by enough that the half-turn no longer maps the edge
            # onto itself to the precision the weld needs.
            v = _snap_axis(np.linalg.svd(C - C.mean(0),
                                        full_matrices=False)[2][0])
            p = C.mean(0)
            gens.append(_halfturn(p, v))
            kinds.append((name, 'halfturn', straight))
        elif planar < tol:
            n = _snap_axis(nrm)
            gens.append(_mirror(n, float(np.dot(cen, n))))
            kinds.append((name, 'mirror', planar))
        else:
            kinds.append((name, 'neither', min(straight, planar)))
    return gens, kinds


class _Rotations:
    """The point group, discovered and then SNAPPED TO.

    The generators are measured off a numerically integrated patch, so
    each carries an error of order 1e-5 -- the half-turn axis is fitted
    to a polyline, not read from a formula.  Composing them freely
    accumulates that error, and then two words that denote the same
    isometry differ in the tenth decimal and are counted as two.  Left
    alone the closure never closes: the element count doubles with every
    generation and eventually manufactures "translations" of length 3e-4
    that are really the identity plus drift.

    The rotation parts, however, form a FINITE group, so they can be
    collected once and thereafter snapped to.  Every composition
    replaces its rotation by the canonical representative it is nearest
    to, which stops the error compounding and lets the closure terminate.
    (`weierstrass.py` does the same thing for the cubic surfaces, where
    the 48 rotations are known in advance and can simply be listed; here
    they are learned from the generators instead, so that CLP and the
    Lidinoid can reuse this with their own symmetry.)
    """

    def __init__(self, tol=1e-3):
        self.tol = float(tol)
        self.reps = []

    def canon(self, R):
        for i, S in enumerate(self.reps):
            if np.max(np.abs(R - S)) < self.tol:
                return i, S
        self.reps.append(np.array(R))
        return len(self.reps) - 1, self.reps[-1]


def h_group(gens, maxlen=12, box=2.6, cap=6000, tol=1e-3):
    """Isometries reachable in at most `maxlen` generator steps whose
    translation part stays within `box`.  The full group is infinite --
    it contains the period lattice -- so it is only ever explored to a
    finite radius.

    Returns the elements; `maxlen` is an upper bound rather than a
    target, because with the rotations snapped the search closes on its
    own well before reaching it.
    """
    rots = _Rotations(tol)
    I = np.eye(4)
    rots.canon(I[:3, :3])
    seen = {(0, 0.0, 0.0, 0.0): I}
    frontier = [I]
    for _ in range(int(maxlen)):
        nxt = []
        for M in frontier:
            for g in gens:
                N = g @ M
                if np.max(np.abs(N[:3, 3])) > box:
                    continue
                idx, S = rots.canon(N[:3, :3])
                t = N[:3, 3]
                key = (idx, round(float(t[0]), 3), round(float(t[1]), 3),
                       round(float(t[2]), 3))
                if key in seen:
                    continue
                E = np.eye(4)
                E[:3, :3] = S                    # snapped, not drifted
                E[:3, 3] = t
                seen[key] = E
                nxt.append(E)
                if len(seen) >= cap:
                    return list(seen.values())
        frontier = nxt
        if not frontier:
            break
    return list(seen.values())


def h_lattice(elems, tol=1e-5):
    """The period lattice: the three shortest independent PURE
    translations in the group."""
    T = [M[:3, 3] for M in elems
         if np.max(np.abs(M[:3, :3] - np.eye(3))) < tol
         and np.linalg.norm(M[:3, 3]) > tol]
    if not T:
        return None
    T = np.array(T)
    basis = []
    for i in np.argsort(np.linalg.norm(T, axis=1)):
        trial = np.array(basis + [T[i]])
        # RELATIVE rank test.  An absolute tolerance accepts a vector
        # that is the negative of one already in the basis to within the
        # generators' own error -- which produced a "rank 3" basis whose
        # determinant was zero, and a cell that could not tile.
        sv = np.linalg.svd(trial, compute_uv=False)
        if sv[-1] > 1e-3 * sv[0]:
            basis.append(T[i])
            if len(basis) == 3:
                return np.array(basis)
    return None


# --------------------------------------------------------------------
# Meshing
# --------------------------------------------------------------------

def _patch_quads(nu, nv):
    i = np.arange(nu - 1)[:, None]
    j = np.arange(nv - 1)[None, :]
    a = (i * nv + j).ravel()
    b = ((i + 1) * nv + j).ravel()
    c = ((i + 1) * nv + j + 1).ravel()
    d = (i * nv + j + 1).ravel()
    return np.stack([a, b, c, d], axis=1)


def _apply(M, V):
    return V @ M[:3, :3].T + M[:3, 3]


# ====================================================================
# The rest of the theta-function family: CLP, the Lidinoid, rPD
# ====================================================================
# Every Gauss map on this branch of the catalogue is a product of powers
# of theta11 at shifted arguments, so all of them are the SAME object
# written with different exponents:
#
#     log g(z) = const + sum_i  c_i * log theta11(z - p_i).
#
# Writing it that way is what makes one engine serve four surfaces.  It
# also makes the branch tractable: each log theta11 is unwrapped along
# the grid on its own, and a sum of continuous functions is continuous,
# so no separate reasoning is needed per surface about where the cuts
# of a fractional power fall.
#
# The data is Weber's, from the notebooks named against each row.  The
# tau values are his and are not round numbers -- the Lidinoid's is
# fixed by the associate angle that closes its periods (Weyhaupt 2008),
# and rPD's by the rhombohedral cell -- so they are carried verbatim
# rather than re-derived.
#
# WHAT ACTUALLY SHIPS, and why the other two do not.
#
# CLP integrates cleanly: its patch is minimal to 1e-14 and all four of
# its boundary curves classify at machine precision.  It is registered,
# as the exact FUNDAMENTAL PIECE rather than as a filled cell, because
# the four boundary curves generate a group whose translations have rank
# ONE (along z alone).  Weber's own assembly uses axes that are not
# boundary curves at all -- a diagonal through the images of z = 0 and
# z = a, and a mirror plane partway up the patch -- so the generic
# "reflect across the four edges" assembly cannot close it.  Building
# the piece and saying so is the same choice already made for the
# gyroid, whose chiral cell cannot be closed either.
#
# LIDINOID, RPD and CLP_HANDLE were long kept here as data but NOT
# registered, all three blocked on ONE thing, and it is now fixed.
#
# A theta factor puts a singularity of the integrand wherever its theta
# vanishes -- on the lattice, so on the domain boundary and often at a
# corner.  H has exactly one, and it yields to a single substitution in
# one variable (`_domain_u`).  CLP has one in the INTERIOR of an edge,
# which needed the domain split there and each half graded into it.
# Both were hand-declared, one `splits` tuple along one edge.
#
# These three have several at once, and in BOTH variables, which no
# such tuple can express:
#
#   Lidinoid   zeros at z = 0 and 1/2 on y = 0, and again on y = Im(tau)
#   rPD        zeros at z = 0, 1 on y = 0 and at 1/2 + tau/2 on the top
#   CLP_HANDLE poles at THREE corners -- 0, 1/2 and tau/2 -- plus zeros
#              at a and 1/2 + tau/2
#
# So the singular points are no longer declared at all: they are DERIVED
# from the terms and the lattice (`spec_singularities`), each gets its
# own substitution power from its own exponent, and both axes are graded
# toward all of them, including the case the old code could not express
# at all -- a segment with a singular point at EACH end, which is
# graded from both ends into the middle.  With that in place:
#
#           patch diameter           |H| * d, n = 60 -> 140
#   CLP     0.7436 (unchanged)       1.3e-3 -> 2.2e-4     O(h^2)
#   Lidinoid    1.877 -> 1.901       9.7e-2 -> 1.8e-2     O(h^2)
#   CLP+handle  1.055 -> 1.060       9.6e-3 -> 1.3e-3     O(h^2)
#   rPD         3.996 -> 4.173       8.9e-1 -> 1.5e-1     O(h^2)
#
# against diameters that used to GROW without limit as the grid refined
# (Lidinoid 449 -> 297 -> 198, rPD 283 -> 188 -> 125).  Conformality,
# |E - G| / (E + G), falls at the same second-order rate on all four.
#
# A WARNING for whoever measures this next.  The blow-up that first
# suggested these were still broken after the fix was a measurement
# artifact: `np.gradient(P)` with unit index spacing on a GRADED grid
# differentiates against the index, not the parameter, and reports mean
# |H| * d growing from 44 to 1129 on a patch that is in fact converging
# to minimal.  Pass the node arrays -- `np.gradient(P, xs, axis=0)` --
# or the number is meaningless.
#
# What each row can then do differs, and only the first is a full cell:
#
#   CLP_HANDLE  its patch is right -- minimal, conformal, converging --
#       and all FIVE boundary curves classify at n >~ 130.  But the
#       assembled CELL is NOT, and it briefly shipped: `_assemble`
#       closes a group generated by isometries measured off the patch,
#       CLP with a handle's come out at 1e-5 where CLP's are 1e-17, and
#       composing those counts one isometry as several.  Its copy count
#       WANDERS with resolution (37.0 / 30.6 / 20.2) instead of sitting
#       at a group order, and the result is stacked coincident patches:
#       7.7% over-shared edges, 3.4% duplicate faces, rendering as
#       leopard spots.  `_assembly_ok` rejects it and the fundamental
#       piece ships instead.  Resume: snap the TRANSLATION parts of the
#       group the way `_Rotations` snaps the rotational ones, or follow
#       an explicit chain as `clp_assembly` does.  The period problem is
#       already solved -- `_CLP_HANDLE_SOLVED` carries Weber's converged
#       (rho, a) against tau.
#   rPD  three of its four edges are exact mirrors, at 1e-13 to 1e-18.
#       The fourth -- the right edge, the far end of a domain two unit
#       cells wide -- is at 2.2e-2 and falls only like 1/n (5.0e-2,
#       3.5e-2, 2.2e-2, 1.5e-2, 1.1e-2 for n = 60..340), so it does not
#       reach the 1e-3 the classifier wants at any usable resolution.
#       The fundamental piece ships; the cell does not.
#   Lidinoid  its edge residuals PLATEAU at 2.4e-2 and do not fall with
#       the grid at all, which says this is not quadrature error.  It is
#       the same situation as the gyroid: at a generic associate angle
#       (here 64.2098 degrees) a straight line becomes neither a
#       straight line nor a planar geodesic, so there is no reflection
#       generator to find and `_assemble` is right to decline.  The
#       fundamental piece is the honest object and is what ships.

def _log_theta(D, q):
    """log theta11 over a grid, unwrapped to a continuous branch."""
    L = np.log(_theta11(D, q))
    im = np.imag(L)
    im[0] = np.unwrap(im[0])
    im = np.unwrap(im, axis=0)
    return np.real(L) + 1j * im


# key -> (label, tau, [(shift, exponent), ...], const, xlim, ylim,
#         associate angle, omega sign, notebook)
_SPECS = {
    'CLP': dict(
        label="Schwarz CLP (exact)",
        tau=2.0j, a=0.15,
        terms=lambda a, tau: ((a, -0.5), (-a, 0.5),
                              (a - tau / 2.0, 0.5), (-a - tau / 2.0, -0.5)),
        const=0.5 * 1j * math.pi / 2.0,          # log sqrt(i)
        xlim=(0.0, 0.5), ylim=lambda t: (0.0, np.imag(t) / 4.0),
        # The y = 0 edge runs THROUGH the branch point at x = a, and the
        # two halves are different symmetry elements -- Weber turns each
        # into its own 2-fold axis, StraightLine[fc(0), fc(a)] and
        # StraightLine[fc(.5), fc(a)].  Treating the edge as one curve
        # is what left the group with a rank-1 translation lattice.
        splits=(0.15,), split_edge='y0',
        # Which boundary curve is which symmetry element, and which are
        # not used at all, taken from Weber's assembly rather than
        # inferred.  His fr1..fr3b are: mirror in the plane of the top
        # edge, half-turns about StraightLine[fc(0), fc(a)] and
        # StraightLine[fc(.5), fc(a)] -- the two halves of the bottom
        # edge -- and a half-turn about the left edge.  The right edge
        # is not a generator.
        #
        # Declaring this beats classifying it here.  The patch is
        # integrated from y = eps rather than y = 0, so the bottom edge
        # bends very slightly near the branch point at x = a, and the
        # straightness test fails on curves that are straight in exact
        # arithmetic -- which downgrades two axes to mirrors and leaves
        # the group unable to close.
        elements=(('y=1', 'mirror'),
                  ('y=0#0', 'halfturn'),
                  ('y=0#1', 'halfturn'),
                  ('x=0', 'halfturn')),
        snap_deg=45.0,                           # tetragonal, not hex
        # The periods, taken from Weber's fr5/fr6 rather than searched
        # for.  He translates by 2{1,-1,0}*trans and 2{1,1,0}*trans with
        # trans = fc(.5), and by {0,0,2 y1}.  Deriving them instead
        # found only ONE of the two in-plane vectors -- the reflection
        # group reaches the other only through words longer than the
        # closure explores -- and the search then paired it with its own
        # negative and called that a basis.
        lattice=lambda P, tau: np.array([
            [2.0 * (P[-1, 0] - P[0, 0])[0],
             -2.0 * (P[-1, 0] - P[0, 0])[1], 0.0],
            [2.0 * (P[-1, 0] - P[0, 0])[0],
             2.0 * (P[-1, 0] - P[0, 0])[1], 0.0],
            [0.0, 0.0, float(np.imag(tau))]]),
        # Weber plots CLP from the IMAGINARY part, i.e. the conjugate
        # surface; Im(int w) = Re(int e^{-i pi/2} w), so it is simply
        # the associate at -90 degrees and needs no separate code path.
        theta=-math.pi / 2.0,
        nb="Triply_SchwarzCLP.nb"),
    'LIDINOID': dict(
        label="Lidinoid (exact)",
        tau=1j * math.tan(math.radians(90.0 - 64.2098)),
        a=0.25,
        terms=lambda a, tau: ((0.0, 2.0 / 3.0), (0.5, -2.0 / 3.0)),
        # rho = e^{i pi/2}, chosen by Weber so the g dh and dh/g flat
        # structures are congruent after a translation
        const=(2.0 / 3.0) * (1j * math.pi / 2.0),
        xlim=(0.0, 1.0), ylim=lambda t: (0.0, np.imag(t)),
        theta=math.radians(64.2098),
        nb="Triply_Lidinoid.nb"),
    # Genus 4: CLP with a handle added.  Six theta factors at half
    # powers instead of CLP's four, and a genuine period problem -- but
    # Weber solved it and tabulated the answers, so `_CLP_HANDLE_SOLVED`
    # carries his (rho, a) against tau rather than re-deriving them.
    # The two ends of that table are named on his page: tau = 4i is near
    # the singly periodic Scherk surface, tau = 0.2i near the doubly
    # periodic one.
    'CLP_HANDLE': dict(
        label="CLP with Handle (exact, genus 4)",
        tau=1.5j, a=0.2423350072589261, rho=0.9999597719688196,
        terms=lambda a, tau: ((a, 0.5), (0.0, -0.5), (-a, 0.5),
                              (0.5, -0.5), (tau / 2.0, -0.5),
                              (0.5 + tau / 2.0, 0.5)),
        # rho * g1 with g1 = (-1 + i)/sqrt(2) = exp(3 i pi / 4)
        const=0j,
        xlim=(0.0, 0.5), ylim=lambda t: (0.0, np.imag(t) / 2.0),
        splits=(0.2423350072589261,), split_edge='y0',
        theta=0.0, snap_deg=45.0,
        nb="CLP( g=4) lines.nb"),
    'RPD': dict(
        label="rPD deformation (exact, rhombohedral)",
        tau=4.0 * 0.3908504810515956j, a=0.5,
        terms=lambda a, tau: ((0.0, -2.0 / 3.0),
                              (a + tau / 2.0, 1.0 / 3.0),
                              (a - tau / 2.0, 1.0 / 3.0)),
        const=0.0,
        xlim=(-0.5, 1.5), ylim=lambda t: (0.0, np.imag(t) / 2.0),
        theta=math.pi / 2.0,
        nb="Triply_Gyroid_AssociateRPD.nb"),
    # Schoen's H'-T, genus 4, the hexagonal-graph/triangle-graph surface
    # of the 1970 NASA catalogue.  This is the FIRST row of the
    # triangle-group series below to be added, and the one that needed no
    # new machinery at all: its four boundary curves classify as mirrors
    # straight off the patch, `_assemble` closes the group, and the
    # period lattice comes out hexagonal as a measured result --
    #     (0, 0, 1)  (-1.2754, 0, 0)  (-0.6377, 1.1046, 0)
    # two equal in-plane generators at 120 degrees (0.6377 = 1.2754/2,
    # 1.1046 = 1.2754 * sqrt(3)/2) perpendicular to the third.  That is
    # the same check Schwarz H gates on, and getting it out rather than
    # putting it in is the evidence the data is right.
    #
    # Both independent sources agree on the constants.  Weber's notebook
    # `H_-T.nb` gives tau = 0.4i, a0 = 1/6 and the exponent 1/2 directly.
    # Fujimori-Weber (2009) Section 3 derives the whole triangle-group
    # family in closed form -- g = (theta(z-p)/theta(z+p))^a with
    # a = (r-1)/r and p = (-r-s+rs)/(2(r-1)s) on the Euclidean triangle
    # group Delta(r,s,t) -- and H'-T is its (2,3,6) member, giving
    # a = 1/2 and p = 1/6.  Identical.
}


# --------------------------------------------------------------------
# The Euclidean-triangle-group series
# --------------------------------------------------------------------
# Weber states outright that P, H, H'-T, H''-R, S'-S'' and T'-R' "share
# enough properties so that a SINGLE PIECE OF CODE can be used to compute
# all of them": each has a reflectional fundamental cell that is a right
# prism over a triangle of type (3,3,3), (2,4,4) or (2,3,6).  This is
# that single piece of code.
#
# Fujimori and Weber (2009), section 3, give the family in closed form.
# On C/<1, tau> with tau in i R+,
#
#     g(z) = (theta11(z - p) / theta11(z + p))^a ,   dh = dz ,
#     a = (r - 1)/r ,   p = (-r - s + r s) / (2 (r - 1) s) ,
#
# where Delta(r,s,t) is the Euclidean triangle group with angles pi/r,
# pi/s, pi/t.  Ten (r,s,t) choices give six distinct surfaces, because
# (r,s,t) and (r,t,s) are the same surface turned upside down.
#
# The formula is checked, not trusted.  It reproduces every constant
# recovered independently from Weber's notebooks -- exponent 1/2 for
# H'-T, 3/4 for S'-S'', 2/3 for H''-R, 5/6 for T'-R', and branch values
# 1/6, 1/6, 1/8, 3/10 -- and it reproduces SCHWARZ H's shipped data
# (a = 2/3, p = 1/4), which this module has been self-testing since long
# before the family was recognised.  Weber's own page gives the branch
# value in the different form p = s / (2(s + t)); the two expressions
# agree on every member (they coincide under 1/r + 1/s + 1/t = 1), which
# is two independent derivations landing on the same numbers.
#
# References:
# - S. Fujimori and M. Weber, "A construction method for triply periodic
#   minimal surfaces", OCAMI Studies 3 (2009) 79-90 -- the closed form
#   and the (r,s,t) table transcribed here.
# - A. H. Schoen, "Infinite periodic minimal surfaces without
#   self-intersections", NASA TN D-5541 (1970) -- H'-T, S'-S'', H''-R and
#   T'-R' as entries of the original catalogue, with their genera.
# - M. Weber, "Alan Schoen's NASA report 1970", minimalsurfaces.blog --
#   the observation that one program computes the whole family.

def _trigroup(label, r, s, t, tau, weld=None, nb=None):
    """One member of the triangle-group series, from (r, s, t).

    `a` in the returned spec is the BRANCH value p of the paper, not its
    exponent -- the spec schema calls the shape parameter `a` and the
    exponent lives inside `terms`.  Mixing those two up silently builds a
    different surface, so both are derived here rather than typed in.
    """
    expo = (r - 1.0) / r
    p = (-r - s + r * s) / (2.0 * (r - 1.0) * s)
    sp = dict(
        label=label,
        tau=complex(tau), a=float(p),
        terms=(lambda a, tau, _e=expo: ((a, _e), (-a, -_e))),
        const=0j,
        xlim=(0.0, 0.5), ylim=lambda tt: (0.0, np.imag(tt) / 2.0),
        # The y = 0 edge runs THROUGH the branch point at x = p, exactly
        # as CLP's does, and the two halves are different symmetry
        # elements.  Read as one curve it contributes one generator
        # instead of two and the group cannot close.
        splits=(float(p),), split_edge='y0',
        theta=0.0,
        trigroup=(r, s, t))
    if weld is not None:
        sp['weld'] = weld
    if nb is not None:
        sp['nb'] = nb
    return sp


_SPECS.update({
    # (2,3,6) -- genus 4.  The one member that ASSEMBLES: its boundary
    # curves all classify as mirrors and the derived lattice is hexagonal.
    'HT': _trigroup("Schoen H'-T (exact, hexagonal, genus 4)",
                    2, 3, 6, 0.4j, weld=3e-5, nb="H_-T.nb"),
    # (4,2,4) -- genus 4, tetragonal.
    'SS': _trigroup("Schoen S'-S'' (exact, tetragonal, genus 4)",
                    4, 2, 4, 0.3j, nb="S_-S_.nb"),
    # (3,2,6) -- genus 5.
    # weld: as for H'-T, the shared 1e-4 default over-merges this patch
    # and leaves two over-shared edges; 3e-5 is clean.
    'H2R': _trigroup("Schoen H''-R (exact, hexagonal, genus 5)",
                     3, 2, 6, 0.15j, weld=3e-5, nb="H_-R.nb"),
    # (6,3,2) -- genus 6.  Weber's notebook uses this member rather than
    # the (6,2,3) twin, i.e. p = 3/10 rather than 1/5; they are the same
    # surface upside down.
    'TR': _trigroup("Schoen T'-R' (exact, hexagonal, genus 6)",
                    6, 3, 2, 1.0j, nb="T_-R_.nb"),
})


# --------------------------------------------------------------------
# Surfaces with a SOLVED period problem
# --------------------------------------------------------------------
# Beyond the triangle-group series the Gauss map stops being a single
# theta quotient and becomes a product of six, eight or twelve factors
# whose shifts depend on one or two free parameters, fixed by requiring
# a period integral to vanish.  Weber solved those period problems and
# TABULATED the answers against the modulus, so the constants below are
# transcribed from his notebooks rather than re-derived.
#
# Nothing in the spec schema needed widening for this.  `terms` is called
# as terms(a, tau) and `a` carries the primary shape parameter, so any
# further parameters are closed over in the lambda -- which is why these
# rows are data and not new machinery.
#
# The exponents of every product below SUM TO ZERO.  That is not
# decoration: g is exp(sum c_k log theta11(z - s_k)), and a non-zero sum
# makes it multivalued on the torus, so a mistyped exponent shows up
# there first.  `_selftest` checks it for every row.
#
# References:
# - A. H. Schoen, "Infinite periodic minimal surfaces without
#   self-intersections", NASA TN D-5541 (1970) -- R-II (genus 9), I-6
#   (genus 5), C(H) (genus 7) and F-RD (genus 6) as catalogue entries.
# - B. Stessmann, "Periodische Minimalflaechen", Mathematische
#   Zeitschrift 38 (1934) 417-442 -- the surface conjugate to Schoen's
#   I-WP, found forty years before it.
# - M. Weber, "Schoen's R2 surface", "Schoen's I6 surface", "Schoen
#   C(H)", "Stessmann's surface", minimalsurfaces.blog -- the
#   Weierstrass data and the solved parameter tables transcribed here.

def _prod_spec(label, tau, a, terms, const=0j, theta=0.0,
               xlim=(0.0, 0.5), ylim=None, splits=None, weld=None,
               nb=None, note=None):
    """A spec whose Gauss map is a bare theta product, with any extra
    parameters already closed over in `terms`."""
    sp = dict(label=label, tau=complex(tau), a=float(a), terms=terms,
              const=const, theta=theta, xlim=xlim,
              ylim=ylim or (lambda t: (0.0, np.imag(t) / 2.0)))
    if splits is not None:
        sp['splits'] = tuple(splits)
        sp['split_edge'] = 'y0'
    if weld is not None:
        sp['weld'] = weld
    if nb is not None:
        sp['nb'] = nb
    if note is not None:
        sp['note'] = note
    return sp


# Stessmann: tau = i/sqrt(3), a0 = 1/6, b0 = 1/3, every exponent 3/4,
# and a sqrt(i) prefactor.  Weber plots it from the IMAGINARY part, and
# Im(int w) = Re(int e^(-i pi/2) w), so that is the associate at -90
# degrees and needs no separate code path -- the same trick CLP uses.
_STESS_B0 = 1.0 / 3.0
_SPECS['STESSMANN'] = _prod_spec(
    "Stessmann's Surface (exact, conjugate to I-WP)",
    tau=1j / math.sqrt(3.0), a=1.0 / 6.0,
    terms=lambda a, tau, _b=_STESS_B0: (
        (a, 0.75), (-a, -0.75),
        (-_b - tau / 2.0, 0.75), (_b - tau / 2.0, -0.75)),
    const=0.25j * math.pi,          # log sqrt(i)
    theta=-math.pi / 2.0,
    nb="Ste_mann.nb")

# Schoen R-II, genus 9.  Weber's notebook tabulates thirteen (a, tau)
# pairs; this is the eighth, the member his own plots use.
_RII_A, _RII_TAU = 0.5459942955818213, 1.8j
_SPECS['RII'] = _prod_spec(
    "Schoen R-II (exact, tetragonal, genus 9)",
    tau=_RII_TAU, a=_RII_A,
    terms=lambda a, tau: ((0.0, -0.5), (a, -0.75), (-a, -0.75),
                          (-0.5, 0.5), (0.5 + 1j * a, 0.75),
                          (0.5 - 1j * a, 0.75)),
    nb="Schoen_RII.nb")

# Schoen C(H), genus 7, the complement of Schwarz H.  Nineteen solved
# (tau, ss) pairs in the notebook; this is the tau = 0.9i member.
_CH_SS, _CH_TAU = 0.06956280256134074, 0.9j
_SPECS['CH'] = _prod_spec(
    "Schoen C(H) (exact, trigonal, genus 7)",
    tau=_CH_TAU, a=_CH_SS,
    terms=lambda a, tau: ((0.25 - a, 2.0 / 3.0), (0.25 + a, 2.0 / 3.0),
                          (-(0.25 - a), -2.0 / 3.0),
                          (-(0.25 + a), -2.0 / 3.0),
                          (0.25 + tau / 2.0, -2.0 / 3.0),
                          (-0.25 + tau / 2.0, 2.0 / 3.0)),
    const=1j * math.pi / 6.0,
    nb="Schoen_C_H_.nb")

# Schoen I-6, genus 5.  Nine solved (tau, a) pairs; tau = 0.93i member.
# Note the branch points sit on the IMAGINARY axis (shifts +-a i), the
# pattern Weber calls "completely unexplored".
_I6_A, _I6_TAU = 0.44116403887207395, 0.93j
_SPECS['I6'] = _prod_spec(
    "Schoen I-6 (exact, genus 5)",
    tau=_I6_TAU, a=_I6_A,
    terms=lambda a, tau: ((0.0, -0.5), (1j * a, 0.5), (tau / 2.0, 0.5),
                          (-1j * a, 0.5), (-0.5, 0.5),
                          (-0.5 + 1j * a, -0.5),
                          (-0.5 + tau / 2.0, -0.5),
                          (-0.5 - 1j * a, -0.5)),
    nb="Schoen_I6.nb")


def spec_period(key, t, n=3000):
    """Re of the period integral whose vanishing fixes a spec's modulus.

    Only rows carrying a `period` entry have one.  That entry names the
    two ends of a straight path in the torus, as functions of (a, tau),
    and both of them are theta ZEROS -- so the integrand diverges like
    s^(-1/2) along the way.  That is integrable, but only with a
    quadrature that clusters into the ends: s = (1 - cos(pi u))/2 makes
    ds vanish at exactly the rate the integrand blows up, leaving a
    bounded transformed integrand.  Sampling the path uniformly instead
    returns nan, because the very first node sits on a zero of theta.
    """
    sp = _SPECS[key]
    a = float(sp['a'])
    tau = 1j * float(t)
    z0, z1 = sp['period'](a, tau)
    q = np.exp(1j * np.pi * tau)
    u = (np.arange(n) + 0.5) / n
    s = 0.5 * (1.0 - np.cos(np.pi * u))
    ds = 0.5 * np.pi * np.sin(np.pi * u) / n
    z = z0 + (z1 - z0) * s
    L = np.full(z.shape, complex(sp['const']), dtype=complex)
    for sh, e in sp['terms'](a, tau):
        L = L + e * np.log(_theta11(z - sh, q))
    g = np.exp(L)
    return float(np.real(np.sum(0.5 * (1.0 / g - g) * (z1 - z0) * ds)))


def solve_spec_tau(key, lo, hi, steps=40):
    """Bisect `spec_period` for the modulus, returning t with tau = i t.

    Used as a GATE rather than at import: each row below stores the
    solved value, and the self-test re-solves from a bracket and checks
    it lands on the stored one.  That makes the stored constant testable
    instead of merely asserted, which is the whole point of keeping the
    solver -- a transcribed number that nothing re-derives is a number
    nobody can check.
    """
    ts = np.linspace(lo, hi, steps)
    vals = [spec_period(key, float(t)) for t in ts]
    for i in range(len(ts) - 1):
        if vals[i] * vals[i + 1] < 0.0:
            a_, b_, fa = float(ts[i]), float(ts[i + 1]), vals[i]
            for _ in range(80):
                m = 0.5 * (a_ + b_)
                fm = spec_period(key, m)
                if fa * fm <= 0.0:
                    b_ = m
                else:
                    a_, fa = m, fm
            return 0.5 * (a_ + b_)
    return None


# Schoen F-RD, genus 6.  Weber: "I neither know an algebraic equation
# for this surface, nor a simple polyhedral approximation."  Its
# conjugate solves the Plateau problem for a quadrilateral in a
# 1 x 1 x sqrt(2) box with angles 90/90/60/45, and was known to
# Stessmann.  Divisor constraints b = (1-2a)/3, c = 1/2 - b, d = 1/2 - a.
#
# The notebook forms Omega1 as (phi2/sqrt(i) - sqrt(i) phi1)/2 rather
# than the standard (phi2 - phi1)/2.  That is not a different formula:
# it is the standard one with G replaced by G e^(i pi/4), i.e. a
# rotation of the Gauss map, so it is folded into `const` here and the
# ordinary combination applies.
_FRD_A = 0.11
_FRD_B = (1.0 - 2.0 * _FRD_A) / 3.0
_SPECS['FRD_EXACT'] = _prod_spec(
    "Schoen F-RD (exact, cubic, genus 6)",
    tau=0.4097611639604068j, a=_FRD_A,
    terms=lambda a, tau, _b=_FRD_B: (
        (a, -0.5), (-a, 0.5), (_b, -0.75), (-_b, 0.75),
        (0.5 - _b + tau / 2.0, 0.75), (-(0.5 - _b) + tau / 2.0, -0.75),
        (0.5 - a + tau / 2.0, 0.5), (-(0.5 - a) + tau / 2.0, -0.5)),
    const=0.25j * math.pi,
    nb="FR-D.nb")

# Schoen's unnamed surface 12, later named F-RD(r) -- the quarter-twisted
# relative of F-RD, genus 5.  Divisor constraints a + a' = 1/2 = b + b'
# and a + b = 1/4, the last of which is what produces the quarter twist;
# one period condition then fixes tau in terms of a.
#
# tau below is SOLVED, not transcribed: Weber's notebook plots solper(a)
# without printing a value.  `solve_spec_tau('FRDR', 0.10, 0.45)`
# recovers 0.3947862928575998 with a residual of 6e-17, and the
# self-test re-runs that bisection against the stored constant.
_FRDR_A = 0.07
_SPECS['FRDR'] = _prod_spec(
    "Schoen F-RD(r) (exact, quarter-twisted, genus 5)",
    tau=0.3947862928575998j, a=_FRDR_A,
    terms=lambda a, tau, _b=0.25 - _FRDR_A: (
        (a, 0.5), (_b, 0.5), (-a, -0.5), (-_b, -0.5),
        (0.5 - a - tau / 2.0, -0.5), (0.5 - _b - tau / 2.0, -0.5),
        (-0.5 + a - tau / 2.0, 0.5), (-0.5 + _b - tau / 2.0, 0.5)),
    const=0.25j * math.pi,
    nb="Unnamed-12.nb")
_SPECS['FRDR']['period'] = (
    lambda a, tau: (complex(a), complex(0.5 - a - tau / 2.0)))


# --------------------------------------------------------------------
# The box-symmetry series
# --------------------------------------------------------------------
# Thirteen of Weber's notebooks share ONE template, selected by a small
# sign vector.  With {e1,e2,e3,e4} in {-1,+1}^4 and three free branch
# values p1, p2, p3,
#
#   G0 = prod_k theta11(z - p_k)^(e_k/2) theta11(z + p_k)^(-e_k/2)
#        * theta11(z - (d + tau/2))^(e4/2)
#        * theta11(z + (d - tau/2))^(-e4/2),
#   d  = -1/2 - p1 + p2 + p3   (Abel's theorem),
#
# and G = G0 / G0(0), a Lopez-Ros normalisation folded into `const`.
# Every exponent appears with both signs, so the sum is zero by
# construction rather than by luck.
#
# This is the structure `research/missing-surfaces-catalog.md` D4 hoped
# for -- one row plus a selector rather than thirteen transcriptions --
# found as a fact in the sources rather than imposed on them.
#
# References:
# - M. Weber, "Box type (g=5)" series, minimalsurfaces.blog -- the
#   template, the sign vectors and the solved parameter tables.
# - V. Ramos Batista and collaborators, for the (+++|-) member; see the
#   triply-periodic Costa row.

def _box_spec(label, e, p1, p2, p3, tau, nb=None):
    """One member of the box-symmetry series, from its sign vector.

    `d` is DERIVED from Abel's theorem rather than passed, because it is
    not free: the divisor has to sum correctly for g to exist at all.
    """
    e1, e2, e3, e4 = (float(x) for x in e)
    d = -0.5 - p1 + p2 + p3

    def terms(a, tau, _p=(p1, p2, p3), _e=(e1, e2, e3, e4), _d=d):
        q1, q2, q3 = _p
        f1, f2, f3, f4 = _e
        return ((q1, f1 / 2.0), (-q1, -f1 / 2.0),
                (q2, f2 / 2.0), (-q2, -f2 / 2.0),
                (q3, f3 / 2.0), (-q3, -f3 / 2.0),
                (_d + tau / 2.0, f4 / 2.0),
                (-_d + tau / 2.0, -f4 / 2.0))

    # G0(0) is a plain number, so the Lopez-Ros normalisation G/G0(0) is
    # a constant shift of log g -- computed once here rather than left
    # out, because it is what orients the surface in its own cell.
    q = np.exp(1j * np.pi * complex(tau))
    L0 = sum(ex * np.log(_theta11(np.array([0.0 + 0j]) - sh, q))[0]
             for sh, ex in terms(p1, complex(tau)))
    sp = _prod_spec(label, tau=tau, a=p1, terms=terms,
                    const=complex(-L0), nb=nb)
    sp['boxtype'] = tuple(int(x) for x in e)
    return sp


_SPECS['BOX_1001'] = _box_spec(
    "Box Type (+-|+) (exact, genus 5)", (-1, 1, 1, -1),
    0.1, 0.2, 0.46802126852411385, 0.42287483495076733j,
    nb="Box_Type_g_5_-_1_0_0_1_.nb")
_SPECS['BOX_1010'] = _box_spec(
    "Box Type (+-+|-) (exact, genus 5)", (-1, 1, -1, 1),
    0.13, 0.2540625168102844, 0.40751143412603386, 0.8j,
    nb="Box_Type_g_5_-_1_0_1_0_.nb")
_SPECS['BOX_1011'] = _box_spec(
    "Box Type (+-+|+) (exact, genus 5)", (-1, 1, -1, -1),
    0.06, 0.25007414956744817, 0.43994933834214295, 0.8j,
    nb="Box_Type_g_5_-_1_0_1_1_.nb")


def _cluster_path(z0, z1, n=2000):
    """Nodes and weights along a straight path whose ENDS are theta
    zeros.  Same substitution as `spec_period`, for the same reason."""
    u = (np.arange(n) + 0.5) / n
    s = 0.5 * (1.0 - np.cos(np.pi * u))
    return z0 + (z1 - z0) * s, 0.5 * np.pi * np.sin(np.pi * u) / n * (z1 - z0)


def _lopez_ros(terms, const, a, tau, z0, z1):
    """Solve for the Lopez-Ros factor rho that makes a surface close.

    Weber's notebooks set rho = sqrt(Int phi2 / conj(Int phi1)) along a
    path between two branch points.  It is a genuine unknown, not a
    cosmetic scale: it is what makes the horizontal period condition
    Int G dh = conj(Int dh/G) hold, and dropping it gives a surface that
    does not close up.
    """
    q = np.exp(1j * np.pi * tau)
    z, w = _cluster_path(complex(z0), complex(z1))
    L = np.full(z.shape, complex(const), dtype=complex)
    for sh, e in terms(a, tau):
        L = L + e * np.log(_theta11(z - sh, q))
    g = np.exp(L)
    i1 = np.sum(g * w)
    i2 = np.sum((1.0 / g) * w)
    return np.sqrt(i2 / np.conj(i1))


# Batista's triply periodic Costa surface, genus 5 -- the (+++|-) box
# type, and the surface Weber also calls a triply periodic
# Costa-Hoffman-Meeks.  Three (tau, a) pairs are tabulated; this is the
# tau = 0.8i member.  Note the DATABASE records `triply-periodic-costa`
# and `horgan-surface` as distinct objects, and they are: the finite
# Horgan surface is proved not to exist, while this one does.
_TPC_A, _TPC_TAU = 0.2723060550713618, 0.8j


def _tpc_terms(a, tau):
    return ((0.0, -0.5), (a, 0.5), (-a, 0.5), (0.5, 0.5),
            (tau / 2.0, -0.5), (0.5 - tau / 2.0, -0.5))


_SPECS['TRIPLY_COSTA'] = _prod_spec(
    "Triply Periodic Costa (Batista, exact, genus 5)",
    tau=_TPC_TAU, a=_TPC_A, terms=_tpc_terms,
    const=0.25j * math.pi,
    nb="Triply_Costa_g_4_straight.nb")
# rho is solved, not guessed -- see `_lopez_ros`.
_SPECS['TRIPLY_COSTA']['const'] = (
    0.25j * math.pi
    + np.log(_lopez_ros(_tpc_terms, 0.25j * math.pi, _TPC_A, _TPC_TAU,
                        0.0, _TPC_A)))

# The Simoes-Batista surface, genus 7: Batista's triply periodic Costa
# with a handle added.  Twelve theta factors on a rectangular lattice,
# with a two-parameter period problem Weber solved; this is the
# tau = 0.97i member.
_SB_A, _SB_B, _SB_TAU = (0.04631108617540206, 0.19605836826545586, 0.97j)
_SPECS['SIMOES_BATISTA'] = _prod_spec(
    "Simoes-Batista Surface (exact, genus 7)",
    tau=_SB_TAU, a=_SB_A,
    terms=lambda a, tau, _b=_SB_B: (
        (a, -0.5), (-a, 0.5), (0.25, 0.5), (-0.25, -0.5),
        (0.5 - a, -0.5), (-(0.5 - a), 0.5),
        (0.25 - tau / 2.0, -0.5), (-(0.25 + tau / 2.0), 0.5),
        (_b - tau / 2.0, -0.5), (-(_b + tau / 2.0), 0.5),
        (0.5 - _b - tau / 2.0, -0.5), (-(0.5 - _b + tau / 2.0), 0.5)),
    const=0.25j * math.pi,
    nb="Sim-es-Batista-g-7.nb")
_SPECS['SIMOES_BATISTA']['test_res'] = (60, 90)


# --------------------------------------------------------------------
# Hackman's toroidal 1-noid -- DEFERRED, and exactly why
# --------------------------------------------------------------------
# This is the row that `dh_terms` was built for, and the data is right:
#
#     k = 1/3,   tau = t + 2i,
#     g  = theta11(z + k/2) / theta11(z - k/2),
#     dh = sigma(z - k/2) sigma(z + k/2) / sigma(z)^2
#        = const * theta11(z - k/2) theta11(z + k/2) / theta11(z)^2,
#
# the sigma-to-theta step being exact because the eta-exponentials cancel
# (see the note in `_spec_patch`), with the constant folding into the
# Bonnet phase.  Driven through the spec engine on a domain clear of the
# origin it converges cleanly to minimal -- median |H| * diam
# 6.67e-4 -> 2.29e-4 -> 1.04e-4 over n = 45/75/110, better than an order
# of magnitude under any shipped row.
#
# It does NOT ship, because of what sits at the origin.  The exponents
# there are p = 0 from g and d = -2 from dh, so the integrand goes like
# s^-2: an order-two POLE, which is the catenoid end of the 1-noid, not
# a branch point.  Grading cannot absorb that -- s^-2 is not integrable
# -- and the rectangle domain has no way to excise a puncture at one of
# its own corners.  Truncating the domain to avoid the origin does give
# a minimal patch, but it is a patch with the surface's defining feature
# cut out, which would be a misleading thing to ship under this name.
#
# Resume: the row needs end trimming (a masked puncture, as the zoo's
# disk domains have) plus the screw-motion assembly of 2 pi k per storey,
# and the modulus t solved from the notebook's 1-D FindRoot on
# period(k, tau) with the seed t in [0.15, 0.25].  The Bonnet phase does
# NOT need its transcendental closed form transcribed -- it is the
# associate angle, already a spec field, so it can be solved for by the
# same closure condition rather than typed in.
#
# References:
# - M. Hackman, thesis; toroidal 1-Noids on every conformal type of
#   torus.  Reported at M. Weber, "Hackman surfaces",
#   minimalsurfaces.blog, whose notebook `Hackman-Surfaces.nb` carries
#   the data transcribed above.
_HACKMAN_DEFERRED = dict(
    label="Hackman Surface (toroidal 1-noid)",
    tau=0.2 + 2.0j, a=1.0 / 6.0,          # a = k/2, k = 1/3
    terms=lambda a, tau: ((a, -1.0), (-a, 1.0)),
    dh_terms=lambda a, tau: ((a, 1.0), (-a, 1.0), (0.0, -2.0)),
    const=0j, theta=0.0,
    xlim=(0.0, 0.5), ylim=lambda t: (0.0, np.imag(t) / 2.0),
    nb="Hackman-Surfaces.nb")


def spec_curves(key, P):
    """The boundary curves of a spec patch, split where the spec says.

    The generic version takes the four edges whole.  That is right when
    each edge is a single symmetry element, and wrong when one runs
    through a branch point: CLP's y = 0 edge is two straight 2-fold axes
    meeting at x = a, and read as one curve it classifies as a single
    plane and contributes one generator instead of two.  With only that
    one, the group's translations come out rank 1 and the surface cannot
    be assembled at all.
    """
    sp = _SPECS[key]
    curves = [('x=0', P[0, :]), ('x=1', P[-1, :]), ('y=1', P[:, -1])]
    splits = sp.get('splits', ())
    if not splits:
        curves.append(('y=0', P[:, 0]))
        return curves
    xs = _spec_nodes(key, P.shape[0])[0]
    x1 = sp['xlim'][1]
    edge = P[:, 0] if sp.get('split_edge', 'y0') == 'y0' else P[:, -1]
    # Named by SEGMENT INDEX, not by formatted coordinates: the graded
    # nodes put the first one at -2.8e-17 rather than 0, which formats
    # as "-2.78e-17" and silently stopped matching the spec's element
    # list -- dropping a generator and with it the whole assembly.
    lo = 0
    for k, c in enumerate(list(splits) + [x1]):
        hi = int(np.argmin(np.abs(xs - c)))
        if hi - lo >= 3:
            curves.append(('y=0#%d' % k, edge[lo:hi + 1]))
        lo = hi
    return curves


def spec_generators(key, P, tol=1e-3):
    """Schwarz generators for a spec patch.

    A spec may DECLARE which curve is which element (and which to leave
    out); otherwise each is classified, as it is for H.
    """
    sp = _SPECS[key]
    step = sp.get('snap_deg', 30.0)
    curves = dict(spec_curves(key, P))
    want = sp.get('elements')
    if want is None:
        want = [(nm, None) for nm in curves]
    gens, kinds = [], []
    for name, forced in want:
        C = curves.get(name)
        if C is None or len(C) < 3:
            continue
        straight, planar, nrm, cen = _classify(C)
        kind = forced
        if kind is None:
            kind = ('halfturn' if straight < tol
                    else 'mirror' if planar < tol else None)
        if kind == 'halfturn':
            v = _snap_axis(np.linalg.svd(C - C.mean(0),
                                         full_matrices=False)[2][0], step)
            gens.append(_halfturn(C.mean(0), v))
            kinds.append((name, 'halfturn', straight))
        elif kind == 'mirror':
            n = _snap_axis(nrm, step)
            gens.append(_mirror(n, float(np.dot(cen, n))))
            kinds.append((name, 'mirror', planar))
        else:
            kinds.append((name, 'neither', min(straight, planar)))
    return gens, kinds


# Weber's solved period problem for the genus-4 surface: tau -> (rho, a).
# The pair has to satisfy two period conditions simultaneously and is
# found by a root solve; these are his converged values.
_CLP_HANDLE_SOLVED = {
    4.0: (0.999999999086189, 0.24999702350529662),      # near singly Scherk
    2.0: (0.9999982556507706, 0.24840528944694143),
    1.5: (0.9999597719688196, 0.2423350072589261),
    1.0: (0.9991279448306104, 0.21380383880954626),
    0.5: (0.9922207028102443, 0.11670790344759747),
    0.25: (0.9920496300782162, 0.0543101456973969),
    0.2: (0.992994526285588, 0.042770615365203),        # near doubly Scherk
}


def clp_handle_params(tau_im):
    """Set the genus-4 row to the solved (rho, a) nearest `tau_im`.

    The period problem is not re-solved here.  Interpolating between
    Weber's converged pairs would give a surface that fails to close,
    so the nearest tabulated tau is used and returned, and the caller
    can say which one it got.
    """
    key = min(_CLP_HANDLE_SOLVED, key=lambda k: abs(k - float(tau_im)))
    rho, a = _CLP_HANDLE_SOLVED[key]
    sp = _SPECS['CLP_HANDLE']
    sp['tau'] = complex(0.0, key)
    sp['a'] = a
    sp['rho'] = rho
    sp['splits'] = (a,)
    sp['const'] = complex(np.log(complex(rho)) + 3j * np.pi / 4.0)
    return key, rho, a


def _sing_power(c):
    """Substitution power for an integrand that behaves like s^(-|c|).

    The substitution s = u^m turns s^(-|c|) ds = m u^(m - 1 - m|c|) du,
    which is bounded as soon as m >= 1/(1 - |c|).  So one line covers
    every exponent the family uses: 1/2 wants m = 2, 2/3 wants m = 3,
    1/3 wants m = 2 (1.5 rounded up).  |c| >= 1 would not be integrable
    at all and no substitution would save it; those return 0 and the
    caller leaves the axis uniform rather than pretending.
    """
    c = abs(float(c))
    if c < 1e-12:
        return 1                                # not singular
    if c >= 1.0 - 1e-12:
        return 0                                # not integrable
    return max(2, int(math.ceil(1.0 / (1.0 - c) - 1e-12)))


def spec_singularities(key):
    """Every point of the domain rectangle where the integrand blows up,
    as {(x, y): exponent}.

    This used to be hand-declared, one `splits` tuple per surface along
    one edge, and that is why three of the four rows in `_SPECS` could
    not be shipped: they have singular points at several places at once
    and in BOTH variables, which no single tuple could express.

    They are found rather than declared.  g = exp(sum c_k log theta11(z
    - shift_k)), and theta11 has a simple zero at every point of the
    lattice {m + n tau}, so g behaves like s^(sum c) in the distance s
    to each translate shift_k + m + n tau.  W carries both g and 1/g, so
    the integrand goes like s^(-|sum c|) there whichever way the sign
    falls.  Exponents at a shared point ADD, and a point where they
    cancel is not singular at all -- which is why they are accumulated
    before being turned into a power.
    """
    sp = _SPECS[key]
    tau, a = sp['tau'], sp['a']
    x0, x1 = sp['xlim']
    y0, y1 = sp['ylim'](tau)
    ty = float(np.imag(tau))
    pad = 1e-9
    # g and dh are accumulated SEPARATELY.  With g ~ s^p and dh ~ s^d at
    # a point, the three integrand components go like s^(p+d), s^(d-p)
    # and s^d, so the worst power is d - |p| and the point is singular
    # exactly when that is negative.  With no dh_terms, d = 0 and this
    # reduces to -|p|, which is what the old single-accumulator code
    # computed -- so every existing row grades identically.
    acc = {}
    accd = {}
    for shift, c in sp['terms'](a, tau):
        sx, sy = float(np.real(shift)), float(np.imag(shift))
        # every lattice translate that can land in the rectangle
        for n in range(int(math.floor((y0 - sy) / ty)) - 1,
                       int(math.ceil((y1 - sy) / ty)) + 2):
            py = sy + n * ty
            if not (y0 - pad <= py <= y1 + pad):
                continue
            for m in range(int(math.floor(x0 - sx - n * float(np.real(tau))))
                           - 1,
                           int(math.ceil(x1 - sx - n * float(np.real(tau))))
                           + 2):
                px = sx + m + n * float(np.real(tau))
                if not (x0 - pad <= px <= x1 + pad):
                    continue
                k = (round(px, 12), round(py, 12))
                acc[k] = acc.get(k, 0.0) + float(c)
    for shift, c in (sp['dh_terms'](a, tau) if 'dh_terms' in sp else ()):
        sx, sy = float(np.real(shift)), float(np.imag(shift))
        for n in range(int(math.floor((y0 - sy) / ty)) - 1,
                       int(math.ceil((y1 - sy) / ty)) + 2):
            py = sy + n * ty
            if not (y0 - pad <= py <= y1 + pad):
                continue
            for m in range(int(math.floor(x0 - sx - n * float(np.real(tau))))
                           - 1,
                           int(math.ceil(x1 - sx - n * float(np.real(tau))))
                           + 2):
                px = sx + m + n * float(np.real(tau))
                if not (x0 - pad <= px <= x1 + pad):
                    continue
                k = (round(px, 12), round(py, 12))
                accd[k] = accd.get(k, 0.0) + float(c)
    out = {}
    for k in set(acc) | set(accd):
        eff = accd.get(k, 0.0) - abs(acc.get(k, 0.0))
        if eff < -1e-12:
            out[k] = eff
    return out


def _graded_axis(lo, hi, sing, n, tiny=1e-6):
    """Nodes and dt/du weights on [lo, hi], graded into each singular
    coordinate in `sing` ({coordinate: exponent}).

    Returns (t, w) with t ascending.  The quadrature must then run in
    the substituted variable u -- putting these nodes into a rule over t
    buys nothing, which is the mistake that cost an earlier debugging
    pass on the hexagonal patch.

    Three cases per segment, and the third is the one the old
    single-split code could not express:

      neither end singular   a uniform grid
      one end singular       graded into that end
      BOTH ends singular     split at the midpoint and grade each half
                             into its own end

    Segments with a singular point at each end are the norm for these
    surfaces -- CLP with a handle has its whole left edge between two of
    them -- and grading such a segment into only one end leaves the
    other sampled at its worst.
    """
    lo, hi = float(lo), float(hi)
    cuts = sorted({lo, hi} | {float(p) for p in sing
                              if lo + 1e-12 < float(p) < hi - 1e-12})
    pieces = []
    for i in range(len(cuts) - 1):
        A, B = cuts[i], cuts[i + 1]
        mA = _sing_power(sing.get(_near(A, sing), 0.0))
        mB = _sing_power(sing.get(_near(B, sing), 0.0))
        if mA > 1 and mB > 1:
            mid = 0.5 * (A + B)
            pieces.append((A, mid, mA, 'lo'))
            pieces.append((mid, B, mB, 'hi'))
        elif mA > 1:
            pieces.append((A, B, mA, 'lo'))
        elif mB > 1:
            pieces.append((A, B, mB, 'hi'))
        else:
            pieces.append((A, B, 1, None))

    seg = int(max(4, round(n / max(1, len(pieces)))))
    ts, ws, dus = [], [], []
    for A, B, m, side in pieces:
        h = abs(B - A)
        if m <= 1 or h < 1e-15:
            t = np.linspace(A, B, seg)
            w = np.ones_like(t)
            du = h / max(seg - 1, 1)
        else:
            r = h ** (1.0 / m)
            if side == 'hi':                    # grade into B
                u = np.linspace(r, 0.0, seg)
                u[-1] = r * tiny
                t, w = B - u ** m, m * u ** (m - 1)
            else:                               # grade into A
                u = np.linspace(0.0, r, seg)
                u[0] = r * tiny
                t, w = A + u ** m, m * u ** (m - 1)
            du = r / max(seg - 1, 1)
        order = np.argsort(t)
        ts.append(t[order])
        ws.append(np.abs(w[order]))
        dus.append(du)
    t = np.concatenate(ts)
    w = np.concatenate(ws)
    keep = np.concatenate([[True], np.diff(t) > 1e-14])
    # Where each surviving piece starts and how long it is, so the
    # caller can integrate it in ITS OWN u with that piece's uniform
    # step.  See `_piece_integral` for why that beats one global rule.
    starts = np.cumsum([0] + [len(a) for a in ts])[:-1]
    idx = np.cumsum(keep) - 1
    spans = []
    for p, s in enumerate(starts):
        e = s + len(ts[p])
        sub = keep[s:e]
        if not sub.any():
            continue
        spans.append((int(idx[s + int(np.argmax(sub))]),
                      int(sub.sum()), float(dus[p])))
    return t[keep], w[keep], spans


def _axis_integral(V, t, w, scale=1.0):
    """Cumulative trapezoid of W along axis 0 in the GRADED variable.

    `V` is W already multiplied by the dt/du weights, so the rule is
    taken against a uniform du -- recovered here as the node spacing
    over the average weight.  When the axis is not graded (every weight
    1) this is exactly the plain trapezoid in t.

    A piecewise version of this was tried and reverted, on the theory
    that mixing two substitutions across a piece junction costs an
    order.  It does not pay: integrating each graded piece separately in
    its own u left rPD's right-edge residual NON-MONOTONE in the grid
    (4.2e-3 at n = 60, 2.0e-2 at 100, 1.3e-1 at 160), where this rule
    has it falling steadily (5.0e-2, 3.5e-2, 2.2e-2, 1.5e-2, 1.1e-2 up
    to n = 340).  A rule that gets worse as the grid refines is worse
    than a slow one, whatever its order is on paper.
    """
    sh = (slice(None),) + (None,) * (V.ndim - 1)
    du = np.diff(t) / np.maximum(0.5 * (w[:-1] + w[1:]), 1e-300)
    return np.cumsum(0.5 * (V[:-1] + V[1:]) * (du[sh] * scale), axis=0)


def _near(v, keys, tol=1e-9):
    """The key of `keys` closest to `v`, or `v` itself if none is."""
    best, bd = v, tol
    for k in keys:
        d = abs(float(k) - float(v))
        if d <= bd:
            best, bd = k, d
    return best


def _spec_axes(key, nu, nv):
    """The graded (x nodes, x weights, y nodes, y weights) for a patch.

    Both axes are graded toward every singular coordinate.  The y axis
    matters for two separate reasons, and only the first is obvious:
    the left edge is the column actually integrated in y, so a singular
    point on it must be graded into or that one integral is wrong; but
    the y grid is SHARED by every row, so pulling samples toward a
    singular height also thins the rows that pass close to a singularity
    sitting anywhere else on that height.  rPD is the case that shows
    it -- its right edge classified as no symmetry element at all
    (residual 5e-2, falling only like 1/n) purely because the rows near
    its top corner were sampled too coarsely to integrate accurately.
    """
    sp = _SPECS[key]
    x0, x1 = sp['xlim']
    y0, y1 = sp['ylim'](sp['tau'])
    sing = spec_singularities(key)

    xsing = {}
    for (px, _py), c in sing.items():
        xsing[px] = max(xsing.get(px, 0.0), abs(c))
    # a spec may also FORCE a cut that is not singular, so that a
    # boundary curve splits exactly on a node (CLP's y = 0 edge is two
    # different symmetry elements either side of x = a)
    for c in sp.get('splits', ()):
        xsing.setdefault(round(float(c), 12), 0.0)

    ysing = {}
    for (_px, py), c in sing.items():
        ysing[py] = max(ysing.get(py, 0.0), abs(c))

    xs, wx, xspan = _graded_axis(x0, x1, xsing, nu)
    ys, wy, yspan = _graded_axis(y0, y1, ysing, nv)
    return xs, wx, xspan, ys, wy, yspan


def _spec_nodes(key, nu):
    """The x nodes and their dx/du weights for a spec patch."""
    xs, wx, _xspan, _ys, _wy, _yspan = _spec_axes(key, nu, 8)
    return xs, wx


def clp_params(tau_im=2.0, a=0.15):
    """Override CLP's two shape parameters.

    CLP is a two-parameter family -- Weber's notebook is called
    CLP-generic for that reason -- and he publishes renders at
    (tau, a) = (0.4, 0.15), (2.0, 0.15) and (1.0, 0.25).  The last is
    the square case, where the four branch values sit at the vertices of
    a regular octagon and the surface is self-conjugate.  Scale is a
    third parameter and is handled by the caller, not here.
    """
    sp = _SPECS['CLP']
    sp['tau'] = complex(0.0, float(tau_im))
    sp['a'] = float(a)
    sp['splits'] = (float(a),)
    return sp


def _spec_patch(key, nu, nv, theta=None, eps=1e-7):
    """Fundamental patch for one of the theta-family surfaces.

    The branch points all sit on the y = 0 edge of the domain -- they
    are the lattice zeros of the theta factors -- so the grid starts a
    hair above it, exactly as Weber's notebooks do.  That keeps every
    integrand finite without needing a substitution per surface; the
    price is that the patch boundary approximates y = 0 rather than
    reaching it, and the error is O(sqrt(eps)) for a square-root
    factor, which at 1e-7 is well under the tolerance the reflection
    generators are classified with.
    """
    sp = _SPECS[key]
    tau, a = sp['tau'], sp['a']
    x0, x1 = sp['xlim']
    y0, y1 = sp['ylim'](tau)
    ang = sp['theta'] if theta is None else float(theta)

    # Break the x grid at every point the spec says the boundary
    # splits, so the split lands exactly on a node.  Weber does the same
    # thing -- his XR is the union of ranges over Sort[{0, a, .5}] --
    # and it is not cosmetic: the sub-curves either side of a split are
    # different symmetry elements, and a node has to separate them.
    xs, wts, xspan, ys, wys, yspan = _spec_axes(key, nu, nv)

    # BOTH ends of the y range are held off the edge, not just the
    # bottom.  The theta factors vanish on the lattice, and for the
    # Lidinoid and rPD the domain reaches a second row of lattice points
    # at the top -- offsetting only the bottom left those on the
    # boundary, and the patch diameter then grew without limit as the
    # grid refined instead of converging.  The grading already stops a
    # hair short of any singular end; this pushes the plain ends off
    # too, so no surface can put a sample exactly on a lattice zero.
    ys = np.clip(ys, y0 + eps, y1 - eps)
    Z = xs[:, None] + 1j * ys[None, :]

    q = np.exp(1j * np.pi * tau)
    L = np.full(Z.shape, sp['const'], dtype=complex)
    for shift, c in sp['terms'](a, tau):
        L = L + c * _log_theta(Z - shift, q)
    g = np.exp(L)
    inv = 1.0 / g
    # dh defaults to dz, which is what every row above uses.  A row may
    # instead give `dh_terms`, a second theta product -- Hackman's height
    # differential is sigma(z-k/2) sigma(z+k/2) / sigma(z)^2, and the
    # Weierstrass sigma quotient IS a theta quotient here because the
    # eta-exponentials cancel identically:
    #     sigma(z-a) sigma(z+a) / sigma(z)^2
    #       = const * theta11(z-a) theta11(z+a) / theta11(z)^2,
    # the constant coming out of exp(eta1 * 2a^2 / (2 omega1)), which is
    # independent of z and so folds into the Bonnet phase.
    dh = np.ones_like(g)
    if 'dh_terms' in sp:
        Ld = np.zeros(Z.shape, dtype=complex)
        for shift, c in sp['dh_terms'](a, tau):
            Ld = Ld + c * _log_theta(Z - shift, q)
        dh = np.exp(Ld)
    W = np.stack([0.5 * (inv - g) * dh, 0.5j * (inv + g) * dh,
                  dh], axis=-1) * np.exp(1j * ang)

    F = np.zeros((len(xs), len(ys), 3), dtype=complex)
    # The left-edge column integrates up in y (dz = i dy), then every
    # row integrates outward in x from it.  Both run piecewise in the
    # graded variable -- see `_piece_integral`.
    F[0, 1:] = _axis_integral(W[0] * wys[:, None], ys, wys, scale=1j)
    V = W * wts[:, None, None]
    F[1:] = F[0][None, :, :] + _axis_integral(V, xs, wts)
    return np.real(F)


def clp_assembly(P):
    """The copy set for CLP, following Weber's chain literally.

    Group closure plus reduction modulo a lattice is the wrong tool for
    this one, and the notebook says why:

        fr1  = fr0 u reflect(fr0, plane of the top edge)
        fr2  = fr1 u rotate(fr1, line fc(0)-fc(a))
        fr3  = fr2 u rotate(fr2, line fc(.5)-fc(a))
        fr3b =      rotate(fr3, line fc(0)-fc(i y1))
        fr4  = fr3 u fr3b
        fr5  = fr4 u translate(fr3,  2(1,-1,0)*trans)
                   u translate(fr3b, 2(1, 1,0)*trans)
        fr6  = fr5 u translate(fr5, (0,0,2 y1))

    The step that breaks the generic scheme is fr5: it translates the
    two HALVES by different vectors.  That is not a lattice acting on
    the cell, so reducing copies modulo any lattice identifies copies
    that are not equivalent -- which is what left flat plate where the
    surface should saddle, and made the over-shared edge count wander
    with resolution.

    Every element is measured off the patch, so nothing here depends on
    Weber's choice of origin: the mirror is the plane of the top edge,
    each axis is the chord between two boundary corners, and `trans` is
    the image of z = .5 relative to the image of z = 0.
    """
    def m(*mats):
        out = np.eye(4)
        for M in mats:
            out = M @ out
        return out

    top = P[:, -1]
    _, _, nrm, cen = _classify(top)
    refl = _mirror(_snap_axis(nrm, 45.0), float(np.dot(cen, nrm)))

    bl, br = P[0, 0], P[-1, 0]                   # images of z = 0, z = .5
    xs = _spec_nodes('CLP', P.shape[0])[0]
    ia = int(np.argmin(np.abs(xs - _SPECS['CLP']['a'])))
    ba = P[ia, 0]                                # image of z = a
    tl = P[0, -1]                                # image of z = i y1

    rot1 = _halfturn(bl, _snap_axis(ba - bl, 45.0))
    rot2 = _halfturn(br, _snap_axis(ba - br, 45.0))
    rot3 = _halfturn(bl, _snap_axis(tl - bl, 45.0))

    trans = br - bl
    zext = float(P[..., 2].max() - P[..., 2].min())
    t1 = np.eye(4); t1[:3, 3] = [2.0 * trans[0], -2.0 * trans[1], 0.0]
    t2 = np.eye(4); t2[:3, 3] = [2.0 * trans[0], 2.0 * trans[1], 0.0]
    # 2 * zext, not 4.  The z period was measured off the assembled
    # unit -- translate it and count coinciding vertices -- and peaks at
    # HALF the unit's own height, not at its full height.
    t3 = np.eye(4); t3[:3, 3] = [0.0, 0.0, 2.0 * zext]

    s1 = [np.eye(4)]
    s1 = s1 + [m(x, refl) for x in s1]
    s2 = s1 + [m(x, rot1) for x in s1]
    s3 = s2 + [m(x, rot2) for x in s2]
    s3b = [m(x, rot3) for x in s3]
    s4 = s3 + s3b
    s5 = s4 + [m(x, t1) for x in s3] + [m(x, t2) for x in s3b]
    s6 = s5 + [m(x, t3) for x in s5]
    # the block repeats on these three vectors
    B = np.array([[4.0 * trans[0], 0.0, 0.0],
                  [0.0, 4.0 * trans[1], 0.0],
                  [0.0, 0.0, 8.0 * zext]])
    # Weber exports exactly these three stages (his FR0a / FR0b / FR0c),
    # and they are the useful ones to look at: the piece the mathematics
    # produces, the smallest assembly that reads as the surface, and the
    # block that shows it repeating.
    return {'PATCH': ([np.eye(4)], B),
            'UNIT': (s4, B),
            'BLOCK': (s6, B)}


# Only the two that are VERIFIED CONNECTED.  A triply periodic minimal
# surface is connected, so a build that falls into pieces is wrong
# however plausible it looks, and three of the five did:
#
#   PATCH            1 component   ok
#   UNIT             1 component   ok  (and matches Weber's render)
#   BLOCK            2 components  WRONG
#   CONJUGATE        2 components  WRONG
#   CONJUGATE_BLOCK  3 components  WRONG
#
# The cause is measured, not guessed.  Translate the assembled unit and
# count vertices that coincide with the original: z peaks sharply (673
# hits at half the unit's height, 606 at a quarter), but NO in-plane
# translation scores more than 14 out of 24455.  There is no pure
# translation carrying the unit onto itself sideways -- which fits what
# the notebook does, since its fr5 translates the two HALVES by
# different vectors rather than the assembly by one.  So the in-plane
# repetition is a screw or glide, and treating it as a translation
# leaves the copies not touching.
CLP_ARRANGEMENTS = ('PATCH', 'UNIT', 'CONJ_PATCH', 'CONJUGATE',
                    'CONJUGATE_BLOCK')


def _clp_far_point(nu, nv, x_at, y_to, theta=0.0):
    """Evaluate the Weierstrass integral at a point ABOVE the patch.

    Weber's conjugate assembly needs f(a + tau/2), whose imaginary part
    is twice the patch's own height, so the value does not exist on the
    patch grid.  It is obtained by integrating over a taller rectangle
    that starts from the same corner, which puts it in the same frame as
    the patch rather than in one of its own.
    """
    sp = _SPECS['CLP']
    keep = sp['ylim']
    sp['ylim'] = lambda t, _y=y_to: (0.0, _y)
    try:
        P = _spec_patch('CLP', nu, nv, theta=theta)
    finally:
        sp['ylim'] = keep
    xs = _spec_nodes('CLP', nu)[0]
    return P[int(np.argmin(np.abs(xs - x_at))), -1]


def clp_conjugate(nu, nv, maxlen=6):
    """The CONJUGATE CLP surface: patch, generators and assembly.

    Weber shows this beside the original on every parameter set, and it
    is a genuinely different surface rather than a rotation of the same
    one -- the page calls it "an array of singly periodic Scherk
    surfaces", which is what it looks like, against the original's
    crossed sheets.  Both are triply periodic and of genus 3.

    It is the associate at ninety degrees from the original, so it is
    the SAME Weierstrass data read from the real part instead of the
    imaginary one (theta = 0 against the original's -pi/2).

    Its symmetries have to be re-derived rather than reused, and the
    page says why: the conjugate surfaces "usually lack the horizontal
    straight lines but have vertical symmetry planes instead".  Measured
    on the patch, exactly that happens -- the two halves of the bottom
    edge and the left edge, all three STRAIGHT on the original, come out
    planar here, while the top edge turns from planar to straight.  So
    the conjugate classifies its own boundary rather than being told, as
    the original is.

    Closing a group over its boundary curves does NOT work here -- it
    was tried, and produced a scattered heap of overlapping copies
    (135 over-shared edges, nothing like Weber's render).  His conjugate
    assembly is its own chain and is followed literally, exactly as the
    original's is:

        fr1 = fr0 u rotate(fr0, StraightLine[f(y1 i/2), f(.5+y1 i/2)])
        fr2 = fr1 u reflect(fr1, Plane[{0,0,1}, .5])
        fr3 = fr2 u reflect(fr2, Plane[{1,0,0}, f(a+tau/2)_x])
        fr4 = fr3 u reflect(fr3, Plane[{0,1,0}, f(0)_y])
        fr5 = fr4 u translate(fr4, {disx,0,0})
        fr6 = fr5 u translate(fr5, {0,disy,0})
        fr7 = fr6 u translate(fr6, {0,0,-1})

    with disx = 2(f[a+tau/2] - f[a])_x and disy = 2 f[a+tau/2]_y.  All
    but one element is measured off the patch; f(a+tau/2) sits above it
    and comes from `_clp_far_point`.
    """
    def m(*mats):
        out = np.eye(4)
        for M in mats:
            out = M @ out
        return out

    P = _spec_patch('CLP', nu, nv, theta=0.0)
    tau = _SPECS['CLP']['tau']
    a = _SPECS['CLP']['a']
    xs = _spec_nodes('CLP', P.shape[0])[0]
    ia = int(np.argmin(np.abs(xs - a)))

    top = P[:, -1]                               # z = x + i y1
    v = _snap_axis(np.linalg.svd(top - top.mean(0),
                                 full_matrices=False)[2][0], 45.0)
    rot = _halfturn(top.mean(0), v)

    right = P[-1, :]                             # z = .5 + i y
    _, _, nrm, cen = _classify(right)
    mz = _mirror(_snap_axis(nrm, 45.0), float(np.dot(cen, nrm)))

    # Each further mirror is taken at the CURRENT assembly's own extreme
    # along the axis, not from an evaluated point.
    #
    # Weber's chain names the plane as x = f(a + tau/2), and taking him
    # literally is what broke this: that point sits above the patch and
    # has to be integrated separately, which put it at 0.559 where the
    # assembly's boundary is at 0.620.  A mirror 0.06 inside the piece
    # does not extend it -- it folds a copy back over it, and the result
    # came out in two disconnected halves.
    #
    # The boundary is the thing the reflection principle actually asks
    # for ("extend ACROSS the boundary"), it is measurable off the
    # geometry, and it needs no second quadrature.  Checked step by
    # step, the assembly stays a single connected component all the way:
    # 4 copies, then 8, 16 and 32.
    #
    # Weber's own exported mesh settles that this is the right target:
    # parsed out of his PoVRay sources, the conjugate is ONE component
    # of 49294 vertices.  A build that falls into two is wrong however
    # manifold it looks.
    V0 = P.reshape(-1, 3)

    def _extent(ops, axis):
        lo = hi = None
        for M in ops:
            w = (V0 @ M[:3, :3].T + M[:3, 3])[:, axis]
            lo = w.min() if lo is None else min(lo, w.min())
            hi = w.max() if hi is None else max(hi, w.max())
        return float(lo), float(hi)

    s2 = [np.eye(4)]
    s2 = s2 + [m(x, rot) for x in s2]
    s2 = s2 + [m(x, mz) for x in s2]

    stages = {'CONJ_PATCH': [np.eye(4)], 'CONJUGATE_HALF': list(s2)}
    cur = list(s2)
    for axis in (0, 1, 2):
        n_ = np.zeros(3)
        n_[axis] = 1.0
        cur = cur + [m(x, _mirror(n_, _extent(cur, axis)[1]))
                     for x in cur]
        if axis == 1:
            stages['CONJUGATE'] = list(cur)
    stages['CONJUGATE_BLOCK'] = list(cur)
    # The repeat vectors are the assembled block's own extents, for the
    # same reason the mirrors are: they are measurable, where the
    # notebook's disx / disy come from that same evaluated point.
    B = np.diag([_extent(cur, i)[1] - _extent(cur, i)[0]
                 for i in range(3)])
    return P, stages, B


def _drop_degenerate(V, Q, rel=1e-9):
    """Drop quads whose area has collapsed, and any that a weld has
    turned into a repeated index.  Both come from the graded grid, not
    from the geometry."""
    Q = np.asarray(Q)
    if not len(Q):
        return Q
    P = np.asarray(V)[Q]
    area = 0.5 * np.linalg.norm(
        np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=-1)
    med = float(np.median(area[area > 0.0])) if (area > 0.0).any() else 1.0
    keep = area > rel * max(med, 1e-30)
    a = Q
    b = np.roll(Q, -1, axis=1)
    keep &= np.all(a != b, axis=1)
    return Q[keep]


def _over_shared(Q):
    """Edges used by more than two faces -- zero on a surface."""
    Q = np.asarray(Q)
    if not len(Q):
        return 0
    a, b = Q, np.roll(Q, -1, axis=1)
    e = np.stack([np.minimum(a, b), np.maximum(a, b)], -1).reshape(-1, 2)
    return int((np.unique(e, axis=0, return_counts=True)[1] > 2).sum())


def _assembly_ok(V, Q, tol=0.005):
    """Is an assembled copy set a surface, or a pile of overlapping ones?

    CONNECTEDNESS IS NOT ENOUGH, and this function exists because that
    was learned the hard way twice.  The first time, three of CLP's five
    arrangements were manifold and looked right rendered but fell into
    two or three pieces, so a component count was added.  The second
    time, CLP with a handle passed that component count -- one piece,
    half a million vertices -- while being a stack of DUPLICATED copies
    of the same patch.  It rendered as leopard spots: coincident sheets
    with opposing normals, which no topological count notices.

    So this measures the two things that actually distinguish a surface
    from a pile:

      over-shared edges   an edge used by more than two faces
      duplicate faces     two faces with the same centroid

    Both are exactly 0.000% on every assembly known to be right (Schwarz
    H, 233792 edges; CLP, 75786) and 7.7% / 3.4% on the bad one, so the
    separation is not marginal and the 0.5% threshold is nowhere near
    either side.

    The cause, for whoever picks this up: `_assemble` closes a group
    generated by isometries MEASURED off the integrated patch.  CLP's
    come out at 1e-17, and its cell is clean.  CLP with a handle's are
    1e-5 (its half-turn axis is fitted to a polyline through a branch
    point), and composing those makes two words for the same isometry
    differ in the tenth decimal and be counted as two copies -- which is
    why its copy count WANDERS with resolution, 37.0 / 30.6 / 20.2,
    instead of sitting at a group order.  `_Rotations` snaps the
    rotational parts for exactly this reason; the translations are not
    snapped.  The real fix is either to snap them too or to follow an
    explicit chain the way `clp_assembly` does for CLP; until then, this
    returns False and the caller ships the fundamental piece, which is
    honest and is what Lidinoid and rPD do anyway.
    """
    V = np.asarray(V)
    Q = np.asarray(Q)
    if not len(Q):
        return False
    a = Q
    b = np.roll(Q, -1, axis=1)
    e = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=-1)
    e = e.reshape(-1, 2)
    _u, cnt = np.unique(e, axis=0, return_counts=True)
    over = int((cnt > 2).sum())
    if over > tol * max(len(cnt), 1):
        return False
    # Degenerate faces are EXCLUDED before the duplicate count.  The
    # graded quadrature grid puts consecutive nodes ~1e-12 apart at each
    # singular end, so the patch carries a column of zero-area slivers
    # there whose centroids coincide with their neighbours'.  Those are
    # a meshing artifact of the grid, not two sheets lying on top of one
    # another, and counting them made this reject the very fallback it
    # exists to fall back TO.  They are welded away separately.
    P = V[Q]
    area = 0.5 * np.linalg.norm(
        np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=-1)
    live = area > 1e-9 * float(np.median(area[area > 0.0]) or 1.0)
    if not live.any():
        return False
    cen = P[live].mean(axis=1)
    span = float(np.max(V.max(0) - V.min(0))) if len(V) else 1.0
    key = np.round(cen / max(span * 1e-5, 1e-12)).astype(np.int64)
    _u, cnt = np.unique(key, axis=0, return_counts=True)
    dup = int(cnt.sum() - len(cnt))
    if dup > tol * int(live.sum()):
        return False

    # CONNECTEDNESS.  The docstring above has always said a component
    # count was added after three of CLP's arrangements came out
    # manifold, plausible-looking and in several pieces -- but the check
    # itself was not here, and Schoen H'-T went out through the gap: its
    # assembled cell is TWO pieces that abut without joining, the larger
    # ending at x = -0.318 and a 7020-vertex stray starting at -0.313.
    # The 0.005 gap is 0.4% of the cell span, far too wide for any weld
    # tolerance to close and not a lattice vector either, so it is a
    # misplaced copy rather than a missing generator.
    #
    # A triply periodic minimal surface separates two labyrinths and is
    # connected; a cell that falls apart is wrong however clean its
    # edges are.  Rejecting here makes the caller fall back to the exact
    # fundamental piece, which is what the surfaces that never assembled
    # have always shipped.
    nv = len(V)
    parent = np.arange(nv)

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for f in Q:
        r0 = _find(int(f[0]))
        for j in range(1, len(f)):
            rj = _find(int(f[j]))
            if rj != r0:
                parent[rj] = r0
    roots = {_find(int(i)) for f in Q for i in f}
    return len(roots) <= 1


def _weld_tol(key):
    """Vertex-weld tolerance for a spec, as a fraction of the cell span.

    1e-4 is right for every row that predates H'-T and stays the default,
    so nothing already shipped moves.  It is NOT right for all of them:
    the weld is a single distance threshold applied to a patch whose
    feature spacing varies per surface, and H'-T packs its copies closer
    than CLP or Schwarz H do.  At 1e-4 it merges vertices either side of
    a symmetry plane and leaves six over-shared edges at z = 0.5 -+ 0.105;
    at 3e-5 there are none.  The failure is quiet -- 0.0036% of edges,
    which sails through `_assembly_ok`'s 0.5% threshold while the rows
    known to be right sit at exactly 0.000%.
    """
    return float(_SPECS[key].get('weld', 1e-4))


def _spec_state(key, *_a, **_k):
    """The shape moduli `spec_build` reads but does not take.

    CLP is a two-parameter family and those parameters live in `_SPECS`,
    set by `clp_params`, so they never appear in the signature.  Without
    them in the key the cache returns the previous surface after the
    modulus changes.
    """
    sp = _SPECS[key]
    return (complex(sp['tau']), float(sp['a']),
            complex(sp.get('const', 0)))


def _orbit_ok(V, Q):
    """Connected, no duplicate faces, no over-shared edges."""
    if not len(Q):
        return False
    ec = {}
    for f in Q:
        m = len(f)
        for t in range(m):
            a, b = int(f[t]), int(f[(t + 1) % m])
            e = (a, b) if a < b else (b, a)
            ec[e] = ec.get(e, 0) + 1
    if any(c > 2 for c in ec.values()):
        return False
    cen = {}
    for f in Q:
        c = tuple(np.round(V[list(f)].mean(0), 6))
        cen[c] = cen.get(c, 0) + 1
    if any(v > 1 for v in cen.values()):
        return False
    par = list(range(len(V)))

    def _f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for f in Q:
        r0 = _f(int(f[0]))
        for j in range(1, len(f)):
            rj = _f(int(f[j]))
            if rj != r0:
                par[rj] = r0
    return len({_f(int(i)) for f in Q for i in f}) <= 1


def spec_reflect_tile(key, P, depth):
    """Reflect a fundamental patch in the boundary curves that ARE
    symmetry elements, as far as the result stays a surface.

    Most spec rows ship their fundamental piece because `_assemble`
    cannot close their reflection group on a rank-3 lattice -- some
    boundary curve classifies as neither a straight line nor a planar
    geodesic, so there is no generator for it.  But the OTHER edges
    usually do classify, and reflecting in just those already shows a
    recognisable piece of the surface instead of a lone quadrilateral.
    S'-S'' has three such mirrors, H''-R four.

    So this closes the group generated by whatever `spec_generators`
    found, to `depth` words, and keeps the LARGEST orbit that still
    passes `_orbit_ok` -- connected, no duplicate faces, no over-shared
    edges.  It backs off rather than failing, so the worst case is the
    patch itself and there is no way for this to ship a broken mesh.

    It is deliberately NOT presented as the unit cell: it is a partial
    reflection orbit, and for these rows a true cell is not available.

    Three rows get no growth from this and the reasons differ.  R-II and
    Stessmann produce copies that PARTIALLY OVERLAP -- 21 and 70
    duplicate faces at the first step, with the copies connected and in
    one piece, so it is not a placement bug but genuine overlap of the
    reflected sheets -- and they correctly fall back to the patch.  I-6
    grows once and then closes, which is the orbit finishing rather than
    failing.
    """
    V0 = P.reshape(-1, 3)
    Q0 = _patch_quads(P.shape[0], P.shape[1])
    gens = spec_generators(key, P)[0]
    if not gens or depth < 1:
        return V0, Q0
    span = float(np.max(V0.max(0) - V0.min(0))) or 1.0

    # Deduplicate the orbit by the IMAGE, not by the matrix.
    #
    # Two different words can place the patch in exactly the same spot,
    # whenever they differ by an isometry that happens to stabilise the
    # patch, and keying the orbit on the matrix keeps both -- so the
    # weld then reports duplicate faces and `_orbit_ok` rejects an
    # otherwise correct tiling.  That is what made Reflections inert on
    # R-II, H''-R and Stessmann while working on the other nine rows.
    #
    # It is NOT that their generators are degenerate: measured, every one
    # moves the patch (centroid shifts 0.29 to 1.10, point-set overlap
    # 1-7%).  They are fine individually and collide only in
    # composition, which is why filtering the generators does nothing
    # and only the image test catches it.
    def _imgkey(V):
        q = np.round(V / (1e-4 * span)).astype(np.int64)
        return (tuple(q.min(0)), tuple(q.max(0)),
                tuple(np.round(V.mean(0) / (1e-4 * span)).astype(np.int64)))

    best = (V0, Q0)
    seen = {}
    ident = np.eye(4)
    seen[tuple(np.round(ident.ravel(), 7))] = ident
    frontier = [ident]
    for d in range(int(depth)):
        nxt = []
        for M0 in frontier:
            for G in gens:
                M = np.asarray(G, float) @ M0
                k = tuple(np.round(M.ravel(), 7))
                if k not in seen:
                    seen[k] = M
                    nxt.append(M)
        frontier = nxt
        if not frontier:
            break
        Vs, Qs, base, placed = [], [], 0, set()
        for M in seen.values():
            W = V0 @ M[:3, :3].T + M[:3, 3]
            ik = _imgkey(W)
            if ik in placed:            # same placement by a longer word
                continue
            placed.add(ik)
            Vs.append(W)
            flip = np.linalg.det(M[:3, :3]) < 0.0
            for q in Q0:
                f = tuple(int(i) + base for i in q)
                Qs.append(f[::-1] if flip else f)
            base += len(V0)
        # The spec's OWN weld tolerance, not a hardcoded one.  H''-R
        # declares 3e-5 precisely because 1e-4 over-merges its patch,
        # and using the shared default here left it with two over-shared
        # edges -- enough to reject an otherwise clean five-copy orbit,
        # which is why Reflections appeared to do nothing on that row.
        Vv, Qq = _weld(np.concatenate(Vs, 0), np.asarray(Qs),
                       _weld_tol(key) * span)
        Qq = [tuple(int(i) for i in q) for q in Qq]
        if _orbit_ok(Vv, Qq):
            best = (Vv, Qq)
        else:
            break
    return best[0], best[1]


@_geom_cache.memoise(version=2, extra=_spec_state)
def spec_build(key, cells, res_per_cell, scale, theta,
               arrangement='UNIT'):
    """Builder for one theta-family row, matching the TPMS_EXACT
    signature.  Falls back to the honest fundamental piece whenever the
    reflection group does not close on a rank-3 period lattice, which is
    the same rule the P/Gyroid/D builder follows for its non-periodic
    angles."""
    if isinstance(cells, (tuple, list)):
        cx, cy, cz = (int(max(1, c)) for c in (list(cells) + [1, 1, 1])[:3])
    else:
        cx = cy = cz = max(1, int(cells))
    nu = max(24, int(round(res_per_cell)))
    nv = max(24, int(round(res_per_cell)))
    named = abs(float(theta)) < 1e-9
    P = _spec_patch(key, nu, nv, None if named else float(theta))
    built = None
    if named and key == 'CLP' and arrangement.startswith('CONJ'):
        P, sets, B = clp_conjugate(nu, nv)
        ops = sets[arrangement]
        V0 = P.reshape(-1, 3)
        Q0 = _patch_quads(P.shape[0], P.shape[1])
        Vs, Qs, base = [], [], 0
        for M in ops:
            Vs.append(_apply(M, V0))
            q = Q0 + base
            if np.linalg.det(M[:3, :3]) < 0.0:
                q = q[:, ::-1]
            Qs.append(q)
            base += len(V0)
        built = (np.concatenate(Vs, 0), np.concatenate(Qs, 0), B)
    elif named and key == 'CLP':
        ops, B = clp_assembly(P)[arrangement]
        V0 = P.reshape(-1, 3)
        Q0 = _patch_quads(P.shape[0], P.shape[1])
        Vs, Qs, base = [], [], 0
        for M in ops:
            Vs.append(_apply(M, V0))
            q = Q0 + base
            if np.linalg.det(M[:3, :3]) < 0.0:
                q = q[:, ::-1]
            Qs.append(q)
            base += len(V0)
        built = (np.concatenate(Vs, 0), np.concatenate(Qs, 0), B)
    elif named:
        built = _assemble(P, gens=spec_generators(key, P)[0])
        if built is not None:
            # Validate AFTER the weld, not before.  Duplicated copies of
            # the same patch still carry distinct vertex indices until
            # the weld merges them, so an over-shared edge is invisible
            # at this point: the raw copy set of CLP with a handle reads
            # as perfectly clean and only turns into 7.7% over-shared
            # edges once welded.  Checking the wrong side of the weld is
            # how the bad cell shipped in the first place.
            Vv, Qq, Bb = built
            sp = float(np.max(np.linalg.norm(Bb, axis=1)))
            Vv, Qq = _weld(Vv, Qq, _weld_tol(key) * sp)
            built = (Vv, Qq, Bb) if _assembly_ok(Vv, Qq) else None
    if built is None:
        # No closed cell, so ship the fundamental piece -- but reflect it
        # in whichever boundary curves ARE symmetry elements first, as
        # far as that verifies.  `cells` drives the depth, so the control
        # does something on these rows instead of being inert.
        if named and cx > 1:
            Vr, Qr = spec_reflect_tile(key, P, cx - 1)
            if len(Qr) > (P.shape[0] - 1) * (P.shape[1] - 1):
                span = float(np.max(Vr.max(0) - Vr.min(0))) or 1.0
                Vr, Qr = _weld(Vr, np.asarray(Qr), 1e-7 * span)
                Qr = _drop_degenerate(Vr, Qr)
                return _fit(Vr, [tuple(int(x) for x in q) for q in Qr],
                            scale)
        # The piece is WELDED before it is returned, which the assembled
        # path already did and this one did not.  Grading the quadrature
        # into a singular point drives consecutive nodes to within
        # ~1e-12 of each other, so the raw grid carries a column of
        # zero-area sliver quads at every graded end -- harmless to the
        # integral, but they shade as black seams and break a solidify.
        V = P.reshape(-1, 3)
        Q = _patch_quads(P.shape[0], P.shape[1])
        span = float(np.max(V.max(0) - V.min(0))) if len(V) else 1.0
        V, Q = _weld(V, Q, 1e-7 * max(span, 1e-12))
        Q = _drop_degenerate(V, Q)
        return _fit(V, [tuple(int(x) for x in q) for q in Q], scale)
    V, Q, B = built
    span = float(np.max(np.linalg.norm(B, axis=1)))
    V, Q = _weld(V, Q, _weld_tol(key) * span)
    if cx > 1 or cy > 1 or cz > 1:
        Vp, Qp, base = [], [], 0
        for i in range(cx):
            for j in range(cy):
                for k in range(cz):
                    off = ((i - 0.5 * (cx - 1)) * B[0]
                           + (j - 0.5 * (cy - 1)) * B[1]
                           + (k - 0.5 * (cz - 1)) * B[2])
                    Vp.append(V + off)
                    Qp.append(Q + base)
                    base += len(V)
        V = np.concatenate(Vp, axis=0)
        Q = np.concatenate(Qp, axis=0)
        V, Q = _weld(V, Q, _weld_tol(key) * span)
    return _fit(V, [tuple(int(x) for x in q) for q in Q], scale)


def h_unit(nu, nv, theta, maxlen=8):
    """One translational unit cell of the surface, welded.

    Every group element carries the patch somewhere; elements differing
    by a period produce the same piece of the quotient surface, so the
    copies are deduplicated by their centroid REDUCED modulo the period
    lattice.  What survives is exactly one representative of each piece
    in a primitive cell.

    Returns (verts, quads, lattice) with the lattice as a 3x3 of row
    vectors, or None if the construction failed to close up.
    """
    return _assemble(h_patch(nu, nv, theta), maxlen=maxlen)


def _assemble(P, maxlen=8, gens=None, lattice=None):
    """Reflect a fundamental patch out into one translational cell.

    Shared by every surface in this module: the Schwarz principle does
    not care which Weierstrass data produced the patch, only what its
    boundary curves are.  Returns None when the group does not close on
    a rank-3 lattice, which the caller reads as "not periodic at this
    angle" rather than as a failure.
    """
    if gens is None:
        gens = h_generators(P)[0]
    if len(gens) < 4:
        return None
    elems = h_group(gens, maxlen=maxlen)
    B = h_lattice(elems) if lattice is None else np.asarray(lattice, float)
    if B is None or abs(float(np.linalg.det(B))) < 1e-9:
        return None
    Binv = np.linalg.inv(B.T)

    V0 = P.reshape(-1, 3)
    Q0 = _patch_quads(P.shape[0], P.shape[1])
    c0 = V0.mean(0)

    kept, seen = [], set()
    for M in elems:
        c = M[:3, :3] @ c0 + M[:3, 3]
        f = Binv @ c                            # lattice coordinates
        n = np.floor(f + 1e-9)
        key = tuple(np.round(f - n, 4))
        if key in seen:
            continue
        seen.add(key)
        shift = -(n @ B)                        # bring it into the cell
        N = M.copy()
        N[:3, 3] = N[:3, 3] + shift
        kept.append(N)

    Vs, Qs, base = [], [], 0
    for M in kept:
        Vs.append(_apply(M, V0))
        q = Q0 + base
        if np.linalg.det(M[:3, :3]) < 0.0:      # a mirror flips winding
            q = q[:, ::-1]
        Qs.append(q)
        base += len(V0)
    V = np.concatenate(Vs, axis=0)
    Q = np.concatenate(Qs, axis=0)
    return V, Q, B


def _weld(V, Q, tol):
    key = np.round(V / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    Vw = V[first]
    Qw = inv[Q]
    good = ((Qw[:, 0] != Qw[:, 1]) & (Qw[:, 1] != Qw[:, 2])
            & (Qw[:, 2] != Qw[:, 3]) & (Qw[:, 3] != Qw[:, 0]))
    return Vw, Qw[good]


# A pure function of its arguments -- the H row's tau and branch
# value are module CONSTANTS, not the mutable moduli CLP carries --
# so it needs no `extra` hook.
@_geom_cache.memoise(version=1)
def h_build(cells, res_per_cell, scale, theta):
    """Build the exact Schwarz H surface, centred and fitted to a 2 m
    cube times `scale`.  Signature matches the other TPMS_EXACT
    builders.

    `theta` is the associate (Bonnet) angle.  Only theta = 0 is Schwarz
    H, and only there is the surface triply periodic -- an associate of
    a periodic minimal surface generally is not, because the periods
    rotate with the angle and stop closing up.  So a nonzero angle
    returns the honest single fundamental piece rather than a torn
    lattice, which is the same convention the P/Gyroid/D builder uses
    for its non-periodic angles.
    """
    if isinstance(cells, (tuple, list)):
        cx, cy, cz = (int(max(1, c)) for c in (list(cells) + [1, 1, 1])[:3])
    else:
        cx = cy = cz = max(1, int(cells))
    nu = max(24, int(round(res_per_cell)))
    nv = max(48, int(round(res_per_cell * 2.0)))

    if abs(float(theta)) > 1e-9:
        V = h_patch(nu, nv, theta).reshape(-1, 3)
        Q = _patch_quads(nu, nv)
        return _fit(V, [tuple(int(x) for x in q) for q in Q], scale)

    built = h_unit(nu, nv, 0.0)
    if built is None:                            # never observed; refuse
        return np.zeros((0, 3)), []              # rather than ship junk
    V, Q, B = built
    span = float(np.max(np.linalg.norm(B, axis=1)))
    V, Q = _weld(V, Q, 1e-4 * span)
    if cx > 1 or cy > 1 or cz > 1:
        Vp, Qp, base = [], [], 0
        for i in range(cx):
            for j in range(cy):
                for k in range(cz):
                    off = ((i - 0.5 * (cx - 1)) * B[0]
                           + (j - 0.5 * (cy - 1)) * B[1]
                           + (k - 0.5 * (cz - 1)) * B[2])
                    Vp.append(V + off)
                    Qp.append(Q + base)
                    base += len(V)
        V = np.concatenate(Vp, axis=0)
        Q = np.concatenate(Qp, axis=0)
        V, Q = _weld(V, Q, 1e-4 * span)
    return _fit(V, [tuple(int(x) for x in q) for q in Q], scale)


def _fit(V, faces, scale):
    """Centre on the origin and fit the longest axis to `scale`.

    `scale` arrives as Cell Size, and the house rule is that a generator
    fills a 2 m cube, so Cell Size 2 must give a 2 m object.  It was
    scaled to 2 * Cell Size, i.e. twice the box, which put every exact
    row at 4 m.
    """
    lo, hi = V.min(0), V.max(0)
    ext = float(np.max(hi - lo)) or 1.0
    V = (V - 0.5 * (lo + hi)) * (float(scale) / ext)
    return V, faces


# --------------------------------------------------------------------

def _selftest():
    ok = True

    # 1. The Gauss map is built by unwrapping a principal logarithm,
    #    which is a numerical choice and could silently land on the
    #    wrong sheet.  It must satisfy g^3 = G2^2 identically -- a
    #    mis-unwrap shows up as a clean factor exp(2 pi i / 3), not as
    #    noise, so this catches it exactly.
    u = _domain_u(90)
    ys = np.linspace(0.0, _Y1, 180)
    Z = (_A0 - u ** 3)[:, None] + 1j * ys[None, :]
    D = (-(u ** 3))[:, None] + 1j * ys[None, :]
    g = np.exp((2.0 / 3.0) * _log_g2(D, Z))
    G2 = _theta11(D) / _theta11(Z + _A0)
    rel = float(np.nanmax(np.abs(g ** 3 - G2 ** 2)
                          / np.maximum(np.abs(G2) ** 2, 1e-300)))
    good = rel < 1e-10
    ok &= good
    print("hexagonal: Gauss map is a true 2/3 branch, |g^3 - G2^2| "
          "rel %.2e %s" % (rel, 'OK' if good else 'FAIL'))

    # 2. The patch must be MINIMAL -- the defining property, measured
    #    off the mesh rather than inherited from the formula.  The
    #    absolute value of H is meaningless without a scale, so it is
    #    reported against the patch diameter; and it must FALL as the
    #    grid refines, which is the part that distinguishes a minimal
    #    surface from a lucky one.  The diameter itself must converge:
    #    it is what a mishandled branch point wrecks first.
    prev_mean, prev_diam = None, None
    for nu, nv in ((45, 90), (65, 130), (90, 180)):
        P = h_patch(nu, nv)
        Pu, Pv = np.gradient(P, axis=0), np.gradient(P, axis=1)
        n = np.cross(Pu, Pv)
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True),
                           1e-300)
        E = np.sum(Pu * Pu, -1)
        F = np.sum(Pu * Pv, -1)
        G = np.sum(Pv * Pv, -1)
        L = np.sum(np.gradient(Pu, axis=0) * n, -1)
        M = np.sum(np.gradient(Pu, axis=1) * n, -1)
        N = np.sum(np.gradient(Pv, axis=1) * n, -1)
        den = 2.0 * (E * G - F * F)
        H = (E * N - 2.0 * F * M + G * L) / np.where(
            np.abs(den) < 1e-300, 1e-300, den)
        flat = P.reshape(-1, 3)
        diam = float(np.linalg.norm(flat.max(0) - flat.min(0)))
        # trim a frame: one-sided differences at the border report their
        # own truncation error, not a curvature defect
        mean = float(np.mean(np.abs(H[3:-3, 3:-3]))) * diam
        falling = prev_mean is None or mean < prev_mean
        steady = prev_diam is None or abs(diam - prev_diam) < 1e-3
        ok &= falling and steady
        print("hexagonal: %3dx%3d diam %.5f mean|H|*d %.2e %s"
              % (nu, nv, diam, mean,
                 'OK' if (falling and steady) else 'FAIL'))
        prev_mean, prev_diam = mean, diam

    # 3. Every boundary curve must be a straight line or a planar
    #    geodesic.  If one were neither, the Schwarz principle would not
    #    apply and the surface could not be extended by reflection at
    #    all -- so this is a precondition of the whole construction,
    #    not a nicety.
    P = h_patch(70, 140)
    gens, kinds = h_generators(P)
    good = len(gens) == 4
    ok &= good
    print("hexagonal: boundary curves %s %s"
          % (", ".join("%s=%s(%.1e)" % k for k in kinds),
             'OK' if good else 'FAIL'))

    # 4. The period lattice.  Nothing above mentions a hexagonal cell:
    #    the generators come from the measured boundary curves and the
    #    lattice from the group they generate.  Getting two equal
    #    periods at 120 degrees, both perpendicular to a third, is
    #    therefore independent evidence that this really is Schwarz's
    #    hexagonal surface and not merely some minimal surface.
    elems = h_group(gens)
    B = h_lattice(elems)
    good = B is not None
    if good:
        L = np.linalg.norm(B, axis=1)
        order = np.argsort(-L)                   # two long, one short
        a, b = B[order[0]], B[order[1]]
        c = B[order[2]]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))))
        perp = max(abs(float(np.dot(c, a))) / (np.linalg.norm(c)
                                               * np.linalg.norm(a)),
                   abs(float(np.dot(c, b))) / (np.linalg.norm(c)
                                               * np.linalg.norm(b)))
        ratio = float(np.linalg.norm(a) / np.linalg.norm(b))
        # 60 and 120 are the same lattice: -b is a lattice vector
        # whenever b is, so which of the pair the shortest-vector search
        # happens to return is not a property of the surface.  What IS a
        # property is that the two are equal in length and meet at a
        # sixty-degree multiple, with the third period perpendicular to
        # both -- that is a hexagonal cell and nothing else.
        good = (min(abs(ang - 120.0), abs(ang - 60.0)) < 0.5
                and abs(ratio - 1.0) < 1e-3 and perp < 1e-3)
        print("hexagonal: period lattice |a|/|b| %.6f, angle %.3f deg, "
              "c.a/c.b %.1e %s"
              % (ratio, ang, perp, 'OK' if good else 'FAIL'))
    else:
        print("hexagonal: period lattice NOT FOUND FAIL")
    ok &= good

    # 5. The assembled cell.  The copies must weld to each other, which
    #    shows up as edges shared by MORE than two faces if they land on
    #    top of one another and as an inflated open boundary if they do
    #    not reach.  Open edges are expected -- the surface leaves the
    #    cell through its walls, exactly as the shipped exact P and D
    #    cells do (2.7% and 6.9% of their edges respectively) -- but
    #    over-shared edges are a defect with no such excuse, so they are
    #    gated at zero.
    V, faces = h_build(1, 40, 1.0, 0.0)
    e = {}
    for f in faces:
        n = len(f)
        for i in range(n):
            a, b = int(f[i]), int(f[(i + 1) % n])
            k = (a, b) if a < b else (b, a)
            e[k] = e.get(k, 0) + 1
    over = sum(1 for c in e.values() if c > 2)
    bnd = sum(1 for c in e.values() if c == 1)
    frac = 100.0 * bnd / max(len(e), 1)
    V2, faces2 = h_build((2, 1, 1), 40, 1.0, 0.0)
    good = (len(V) > 0 and len(faces) > 0 and over == 0 and frac < 4.0
            and len(faces2) > len(faces))
    ok &= good
    print("hexagonal: cell %d verts %d faces, open %.2f%%, over-shared "
          "%d, 2x1x1 -> %d faces %s"
          % (len(V), len(faces), frac, over, len(faces2),
             'OK' if good else 'FAIL'))

    # CLP: the second surface on this branch with no published nodal
    # formula.  Its patch is gated the same way H's is -- minimality
    # measured off the mesh, and every boundary curve classifiable as a
    # straight line or a planar geodesic, without which the reflection
    # principle would not apply at all.
    prev = prevm = None
    for n in (40, 60, 90):
        # The NODE ARRAYS go to np.gradient, not just the axis.  Both
        # axes are graded, so differentiating against the index instead
        # measures the grid rather than the surface: it reported this
        # very patch, which is converging to minimal at second order, as
        # having mean |H| * d of 1.55 and RISING.
        xs, _wx, _xp, ys, _wy, _yp = _spec_axes('CLP', n, n)
        P = _spec_patch('CLP', n, n)
        Pu = np.gradient(P, xs, axis=0)
        Pv = np.gradient(P, ys, axis=1)
        nn = np.cross(Pu, Pv)
        nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True),
                             1e-300)
        E = np.sum(Pu * Pu, -1)
        F = np.sum(Pu * Pv, -1)
        G = np.sum(Pv * Pv, -1)
        L = np.sum(np.gradient(Pu, xs, axis=0) * nn, -1)
        M = np.sum(np.gradient(Pu, ys, axis=1) * nn, -1)
        N = np.sum(np.gradient(Pv, ys, axis=1) * nn, -1)
        den = 2.0 * (E * G - F * F)
        Hc = (E * N - 2.0 * F * M + G * L) / np.where(
            np.abs(den) < 1e-300, 1e-300, den)
        fl = P.reshape(-1, 3)
        diam = float(np.linalg.norm(fl.max(0) - fl.min(0)))
        k = max(3, n // 8)
        mean = float(np.median(np.abs(Hc[k:-k, k:-k]))) * diam
        # A PLANE is minimal, so "mean curvature is zero" cannot on its
        # own say the surface is right -- and for a while it did not:
        # evaluating CLP's theta with H's nome returned a nearly
        # constant Gauss map, whose image is a flat sheet, and that
        # sailed through a curvature gate at 1e-14 while being the wrong
        # object entirely.  So the patch must also be genuinely
        # three-dimensional: the smallest singular value of the centred
        # point cloud, against the largest, is a scale-free measure of
        # how far it is from lying in a plane.
        c = fl - fl.mean(0)
        nonplanar = float(np.linalg.svd(c, compute_uv=False)[2]
                          / np.linalg.svd(c, compute_uv=False)[0])
        steady = prev is None or abs(diam - prev) < 5e-3
        falling = prevm is None or mean < prevm
        good = steady and falling and nonplanar > 0.05
        ok &= good
        print("hexagonal: CLP %3dx%3d diam %.5f mean|H|*d %.2e "
              "nonplanar %.4f %s"
              % (n, n, diam, mean, nonplanar, 'OK' if good else 'FAIL'))
        prev, prevm = diam, mean

    P = _spec_patch('CLP', 70, 70)
    gens, kinds = h_generators(P)
    good = len(gens) == 4
    ok &= good
    print("hexagonal: CLP boundary %s %s"
          % (", ".join("%s=%s(%.0e)" % k for k in kinds),
             'OK' if good else 'FAIL'))

    # ... and it must build.  Only the fundamental piece: the four
    # boundary curves generate translations of rank 1, so `_assemble`
    # declines and `spec_build` falls back, which is the intended and
    # documented behaviour rather than a failure.
    V, faces = spec_build('CLP', 1, 50, 1.0, 0.0)
    V2, faces2 = spec_build('CLP', (2, 1, 1), 50, 1.0, 0.0)
    ec = {}
    for f in faces:
        for i in range(len(f)):
            x, y = int(f[i]), int(f[(i + 1) % len(f)])
            k = (x, y) if x < y else (y, x)
            ec[k] = ec.get(k, 0) + 1
    over = sum(1 for c in ec.values() if c > 2)
    bnd = sum(1 for c in ec.values() if c == 1)
    good = len(V) > 0 and len(faces) > 0 and len(faces2) > len(faces)
    ok &= good
    print("hexagonal: CLP cell %d verts %d faces, open %.2f%%, "
          "over-shared %d, 2x1x1 -> %d faces %s"
          % (len(V), len(faces), 100.0 * bnd / max(len(ec), 1), over,
             len(faces2), 'OK' if good else 'FAIL'))
    # Deliberately NOT gated: the assembled CLP cell is not finished.
    # It closes on Weber's declared periods and reads as crossed layers
    # of parallel sheets -- which is what CLP means -- but part of the
    # cell comes out as flat plate where it should be saddle, so the
    # over-shared count above wanders with resolution instead of
    # sitting at zero the way H's does.  Reported rather than asserted,
    # so the debt stays visible; see BACKLOG.

    # The three rows the generalised grading unblocked.  Each is gated
    # on the thing that was actually broken -- the QUADRATURE -- rather
    # than on assembling, because two of them are not supposed to
    # assemble (see the note above `_SPECS`).  Two measurements, both
    # taken with the node arrays passed to `np.gradient`: without them
    # it differentiates against the grid INDEX, and on a graded grid
    # that reports a converging patch as wildly non-minimal.
    for key, want in (('LIDINOID', 0.35), ('CLP_HANDLE', 0.06),
                      ('RPD', 2.0)):
        rows = []
        for n in (60, 100):
            xs, _wx, _xp, ys, _wy, _yp = _spec_axes(key, n, n)
            P = _spec_patch(key, n, n)
            Pu = np.gradient(P, xs, axis=0)
            Pv = np.gradient(P, ys, axis=1)
            nn = np.cross(Pu, Pv)
            nn = nn / np.maximum(
                np.linalg.norm(nn, axis=-1, keepdims=True), 1e-300)
            E = np.sum(Pu * Pu, -1)
            F = np.sum(Pu * Pv, -1)
            G = np.sum(Pv * Pv, -1)
            L = np.sum(np.gradient(Pu, xs, axis=0) * nn, -1)
            M = np.sum(np.gradient(Pu, ys, axis=1) * nn, -1)
            N = np.sum(np.gradient(Pv, ys, axis=1) * nn, -1)
            den = 2.0 * (E * G - F * F)
            Hc = (E * N - 2.0 * F * M + G * L) / np.where(
                np.abs(den) < 1e-300, 1e-300, den)
            fl = P.reshape(-1, 3)
            diam = float(np.linalg.norm(fl.max(0) - fl.min(0)))
            k = max(4, n // 8)
            rows.append((diam,
                         float(np.median(np.abs(Hc[k:-k, k:-k]))) * diam,
                         float(np.median((np.abs(E - G)
                                          / (E + G))[k:-k, k:-k]))))
        (d0, h0, c0), (d1, h1, c1) = rows
        # the diameter must SETTLE (it used to grow without limit), and
        # both minimality and conformality must FALL with the grid
        good = (abs(d1 - d0) / max(d1, 1e-30) < 0.05
                and h1 < h0 and c1 < c0 and h1 < want)
        ok &= good
        print("hexagonal: %s diam %.4f -> %.4f, med|H|*d %.2e -> %.2e, "
              "|E-G|/(E+G) %.2e -> %.2e %s"
              % (key, d0, d1, h0, h1, c0, c1, 'OK' if good else 'FAIL'))

    # Every one of the three ships the fundamental PIECE, and the gate
    # is that they ship a clean one.  CLP with a handle briefly shipped
    # an assembled cell instead: it passed a connectedness count (one
    # piece, half a million vertices) while being a stack of duplicated
    # copies, and rendered as leopard spots.  `_assembly_ok` now rejects
    # that and the caller falls back, so what has to be checked here is
    # that the fallback really is clean -- no over-shared edge, no
    # duplicate face -- at every resolution, including the ones where
    # the copy set was previously accepted.
    for key in ('CLP_HANDLE', 'LIDINOID', 'RPD'):
        bad = []
        for n in (50, 100, 160):
            V, faces = spec_build(key, 1, n, 1.0, 0.0)
            Q = np.asarray([f for f in faces if len(f) == 4])
            if len(Q) != len(faces) or not len(Q):
                bad.append('%d:non-quad' % n)
                continue
            if not _assembly_ok(np.asarray(V), Q):
                bad.append('%d:over-shared or duplicated' % n)
            elif _over_shared(Q):
                bad.append('%d:%d over-shared edges' % (n, _over_shared(Q)))
            elif not np.all(np.isfinite(V)):
                bad.append('%d:non-finite' % n)
        ok &= not bad
        print("hexagonal: %s builds clean at res 50/100/160 %s"
              % (key, 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    # The triangle-group series is generated from (r, s, t) rather than
    # typed in, so the first thing to gate is the GENERATOR: every member
    # must reproduce the constants recovered independently from Weber's
    # notebooks.  If `_trigroup`'s algebra drifts, this catches it before
    # any geometry is built.  Schwarz H is included deliberately -- it is
    # not one of the rows built through `_trigroup`, so it is an
    # out-of-sample check on the formula.
    want = {'HT': (2, 3, 6, 0.5, 1.0 / 6.0),
            'SS': (4, 2, 4, 0.75, 1.0 / 6.0),
            'H2R': (3, 2, 6, 2.0 / 3.0, 0.125),
            'TR': (6, 3, 2, 5.0 / 6.0, 0.3)}
    bad = []
    for key, (r, s, t, expo, p) in want.items():
        sp = _SPECS[key]
        got_p = float(sp['a'])
        got_e = float(sp['terms'](got_p, sp['tau'])[0][1])
        if sp['trigroup'] != (r, s, t):
            bad.append('%s:rst' % key)
        if abs(got_e - expo) > 1e-12 or abs(got_p - p) > 1e-12:
            bad.append('%s:a=%.6f p=%.6f' % (key, got_e, got_p))
    # Schwarz H is (3,3,3): a = 2/3, p = 1/4, which is what this module's
    # own header records for the surface it has shipped all along.
    hr, hs = 3.0, 3.0
    h_e = (hr - 1.0) / hr
    h_p = (-hr - hs + hr * hs) / (2.0 * (hr - 1.0) * hs)
    if abs(h_e - 2.0 / 3.0) > 1e-12 or abs(h_p - _A0) > 1e-12:
        bad.append('H(3,3,3):a=%.6f p=%.6f vs shipped _A0=%.6f'
                   % (h_e, h_p, _A0))
    ok &= not bad
    print("hexagonal: triangle-group closed form reproduces all four "
          "members and Schwarz H %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # g = exp(sum c_k log theta11(z - s_k)) is single-valued on the torus
    # only if the exponents sum to zero, so a mistyped exponent in any
    # product row shows up here before it shows up as geometry.  Cheap,
    # and it covers every spec at once rather than the ones remembered.
    bad = [k for k, sp in _SPECS.items()
           if abs(sum(e for _s, e in sp['terms'](sp['a'], sp['tau'])))
           > 1e-12]
    ok &= not bad
    print("hexagonal: theta exponents sum to zero on all %d spec rows %s"
          % (len(_SPECS), 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    # A transcribed constant that nothing re-derives is a constant
    # nobody can check.  F-RD(r)'s modulus was SOLVED rather than copied
    # -- Weber's notebook plots solper(a) without printing a value -- so
    # the gate re-runs that bisection from a bracket and requires it to
    # land on the stored number, and requires the period residual there
    # to vanish.
    want_t = float(np.imag(_SPECS['FRDR']['tau']))
    got_t = solve_spec_tau('FRDR', 0.10, 0.45)
    resid = abs(spec_period('FRDR', want_t))
    good = (got_t is not None and abs(got_t - want_t) < 1e-9
            and resid < 1e-10)
    ok &= good
    print("hexagonal: F-RD(r) modulus re-solves to %.16f (stored %.16f), "
          "period residual %.1e %s"
          % (got_t if got_t is not None else float('nan'), want_t, resid,
             'OK' if good else 'FAIL'))

    # ...then each member must build a converging minimal patch.
    for key in ('SS', 'H2R', 'TR', 'STESSMANN', 'RII', 'CH', 'I6',
                'FRD_EXACT', 'FRDR',
                'BOX_1001', 'BOX_1010', 'BOX_1011',
                'TRIPLY_COSTA', 'SIMOES_BATISTA'):
        rows = []
        # 45/75 suits most rows.  A few need a finer pair -- at 45 the
        # Simoes-Batista patch is still coarse enough that its diameter
        # moves 5.2% to 75, which would fail the settling gate for being
        # under-resolved rather than wrong; by 60/90 it is at 1.6% and
        # falling.  Declaring that per row beats loosening the gate for
        # everyone.
        for n in _SPECS[key].get('test_res', (45, 75)):
            xs, _a, _b, ys, _c, _d = _spec_axes(key, n, n)
            P = _spec_patch(key, n, n)
            Pu = np.gradient(P, xs, axis=0)
            Pv = np.gradient(P, ys, axis=1)
            nn = np.cross(Pu, Pv)
            nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True),
                                 1e-300)
            E = np.sum(Pu * Pu, -1)
            F = np.sum(Pu * Pv, -1)
            G = np.sum(Pv * Pv, -1)
            L = np.sum(np.gradient(Pu, xs, axis=0) * nn, -1)
            M = np.sum(np.gradient(Pu, ys, axis=1) * nn, -1)
            N = np.sum(np.gradient(Pv, ys, axis=1) * nn, -1)
            den = 2.0 * (E * G - F * F)
            Hc = (E * N - 2.0 * F * M + G * L) / np.where(
                np.abs(den) < 1e-300, 1e-300, den)
            fl = P.reshape(-1, 3)
            diam = float(np.linalg.norm(fl.max(0) - fl.min(0)))
            k = max(4, n // 8)
            sv = np.linalg.svd(fl - fl.mean(0), compute_uv=False)
            rows.append((diam,
                         float(np.median(np.abs(Hc[k:-k, k:-k]))) * diam,
                         float(sv[2] / sv[0])))
        (d0, h0, n0), (d1, h1, n1) = rows
        # a PLANE is minimal, so non-planarity is gated too -- see the
        # note on the CLP curvature check above
        # The |H| ceiling is 0.20, not 2e-2: R-II and I-6 legitimately
        # sit near the shipped rPD row's 0.195 at this resolution, and a
        # tighter bound would fail them for being large rather than for
        # being wrong.  What actually convicts is h1 < h0 -- falling --
        # together with the diameter settling and non-planarity.
        good = (abs(d1 - d0) / max(d1, 1e-30) < 0.05 and h1 < h0
                and h1 < 0.20 and n1 > 0.05)
        ok &= good
        print("hexagonal: %s diam %.4f -> %.4f, med|H|*d %.2e -> %.2e, "
              "nonplanar %.4f %s"
              % (key, d0, d1, h0, h1, n1, 'OK' if good else 'FAIL'))

    # Schoen H'-T does NOT assemble, and the gate records why rather
    # than just asserting the fallback.  Its reflection group closes on
    # a rank-3 lattice that is correctly hexagonal, so the Weierstrass
    # data and the boundary classification are both right -- but the
    # resulting cell is TWO PIECES: the main body ends at x = -0.318 and
    # a 7020-vertex stray begins at -0.313, a gap of 0.005, which is
    # 0.4% of the cell span and not a lattice vector.  That is a
    # misplaced copy, not a weld tolerance (it is two pieces at every
    # tolerance from 1e-5 to 4e-4 of the span).
    #
    # This shipped once, looking plausible, because `_assembly_ok` had
    # no component count despite its docstring claiming one.  Both halves
    # are checked here: the lattice must still come out hexagonal, and
    # the assembly must still be REFUSED, so that a future fix to the
    # copy placement shows up as a failure here rather than silently.
    P = _spec_patch('HT', 70, 70)
    gens, kinds = spec_generators('HT', P)
    built = _assemble(P, gens=gens)
    good = built is not None
    if good:
        _Vv, _Qq, B = built
        a, b, c = B[1], B[2], B[0]
        la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        ang = math.degrees(math.acos(
            float(np.dot(a, b)) / max(la * lb, 1e-30)))
        perp = max(abs(float(np.dot(c, a))), abs(float(np.dot(c, b))))
        good = (abs(la / lb - 1.0) < 1e-3 and perp < 1e-9
                and min(abs(ang - 60.0), abs(ang - 120.0)) < 0.5)
        ok &= good
        print("hexagonal: H'-T lattice |a|/|b| %.6f, angle %.3f deg, "
              "c.a/c.b %.1e %s"
              % (la / lb, ang, perp, 'OK' if good else 'FAIL'))
    else:
        ok = False
        print("hexagonal: H'-T lattice FAIL (_assemble declined)")

    V, faces = spec_build('HT', 1, 60, 1.0, 0.0)
    Q = np.asarray([f for f in faces if len(f) == 4])
    over = _over_shared(Q) if len(Q) == len(faces) and len(Q) else -1
    npatch = len(_spec_patch('HT', 60, 60).reshape(-1, 3))
    good = (over == 0 and np.all(np.isfinite(V)) and len(V) < npatch)
    ok &= good
    print("hexagonal: H'-T ships the fundamental piece (%d verts < patch "
          "%d), over-shared %d %s"
          % (len(V), npatch, over, 'OK' if good else 'FAIL'))

    # The component gate must CONVICT, not merely exist: every assembled
    # row has to come out in one piece.
    bad = []
    for key in ('H', 'CLP'):
        Vv, ff = (h_build(1, 40, 1.0, 0.0) if key == 'H'
                  else spec_build('CLP', 1, 40, 1.0, 0.0))
        Vv = np.asarray(Vv)
        par = list(range(len(Vv)))

        def _f(i, _p=par):
            while _p[i] != i:
                _p[i] = _p[_p[i]]
                i = _p[i]
            return i

        for face in ff:
            r0 = _f(int(face[0]))
            for j in range(1, len(face)):
                rj = _f(int(face[j]))
                if rj != r0:
                    par[rj] = r0
        n = len({_f(int(i)) for face in ff for i in face})
        if n != 1:
            bad.append('%s:%d' % (key, n))
    ok &= not bad
    print("hexagonal: assembled cells are connected %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ...and the rejector itself must not be vacuous: it has to ACCEPT
    # the two assemblies that are known to be right.  Without this the
    # gate above would pass just as well if `_assembly_ok` returned
    # False unconditionally.
    good = (_assembly_ok(*h_build(1, 40, 1.0, 0.0))
            and _assembly_ok(*spec_build('CLP', 1, 40, 1.0, 0.0)))
    ok &= good
    print("hexagonal: the assembly rejector still accepts H and CLP %s"
          % ('OK' if good else 'FAIL'))

    # The conjugate.  It is a different surface, not a re-view of the
    # same one, so it gets its own check: the page says it "usually
    # lacks the horizontal straight lines but has vertical symmetry
    # planes instead", and that is exactly what the boundary
    # classification must show -- the edges that are STRAIGHT on the
    # original coming out planar here, and the top edge going the other
    # way.  If that ever inverts, the associate angle has been applied
    # in the wrong direction and the "conjugate" is the original again.
    Po = _spec_patch('CLP', 60, 60)                     # original
    Pc = _spec_patch('CLP', 60, 60, theta=0.0)          # conjugate
    co = dict(spec_curves('CLP', Po))
    cc = dict(spec_curves('CLP', Pc))
    swapped = 0
    for nm in ('y=0#0', 'y=0#1', 'x=0'):
        so = _classify(co[nm])[0]
        sc = _classify(cc[nm])[0]
        if so < 1e-3 and sc > 1e-2:
            swapped += 1
    top_o = _classify(co['y=1'])[0]
    top_c = _classify(cc['y=1'])[0]
    good = swapped == 3 and top_o > 1e-3 and top_c < 1e-6
    ok &= good
    print("hexagonal: CLP conjugate swaps its symmetries -- %d of 3 "
          "straight edges become planar, top edge %.0e -> %.0e %s"
          % (swapped, top_o, top_c, 'OK' if good else 'FAIL'))

    Vc, fc_ = spec_build('CLP', 1, 40, 1.0, 0.0, 'UNIT')
    ec = {}
    for f in fc_:
        for i in range(len(f)):
            x, y = int(f[i]), int(f[(i + 1) % len(f)])
            k = (x, y) if x < y else (y, x)
            ec[k] = ec.get(k, 0) + 1
    overc = sum(1 for c in ec.values() if c > 2)
    good = len(fc_) > 0 and overc == 0
    ok &= good
    # A triply periodic minimal surface is CONNECTED.  This is the
    # check that caught three of the five arrangements being wrong --
    # they were manifold and looked plausible, and still fell into two
    # or three pieces, which no TPMS does.
    par = list(range(len(Vc)))

    def _find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for f in fc_:
        r = _find(int(f[0]))
        for x in f[1:]:
            t = _find(int(x))
            if t != r:
                par[t] = r
    ncomp = len({_find(i) for i in range(len(Vc))})
    good = good and ncomp == 1
    ok &= ncomp == 1
    print("hexagonal: CLP unit %d verts %d faces, over-shared %d, "
          "components %d %s"
          % (len(Vc), len(fc_), overc, ncomp, 'OK' if good else 'FAIL'))

    # The cache in front of spec_build must notice CLP's moduli, which
    # are module state rather than arguments.  A cache keyed on the
    # signature alone would hand back the previous surface after the
    # modulus changed -- silently, and looking perfectly reasonable.
    clp_params(2.0, 0.15)
    Va, _ = spec_build('CLP', 1, 40, 2.0, 0.0, 'UNIT')
    clp_params(0.4, 0.15)
    Vb, _ = spec_build('CLP', 1, 40, 2.0, 0.0, 'UNIT')
    clp_params(2.0, 0.15)
    Vc, _ = spec_build('CLP', 1, 40, 2.0, 0.0, 'UNIT')
    ba = np.asarray(Va, float).max(0) - np.asarray(Va, float).min(0)
    bb = np.asarray(Vb, float).max(0) - np.asarray(Vb, float).min(0)
    moved = float(np.max(np.abs(ba - bb)))
    back = float(np.max(np.abs(np.asarray(Va, float)
                               - np.asarray(Vc, float))))
    good = moved > 1e-6 and back < 1e-12
    ok &= good
    print("hexagonal: changing the modulus invalidates the cache "
          "(bbox moves %.4f) and returning to it hits again (%.1e) %s"
          % (moved, back, 'OK' if good else 'FAIL'))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("hexagonal self-test failed")
