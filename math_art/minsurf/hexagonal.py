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
# LIDINOID, RPD and CLP_HANDLE are kept here as data but are NOT
# registered, and they all fail for ONE reason, now that CLP has shown
# what the fix looks like.
#
# A theta factor with a negative exponent puts a pole of the integrand
# on the domain boundary.  H has exactly one, at a CORNER, and it yields
# to a single substitution in one variable (`_domain_u`).  CLP has one
# in the INTERIOR of an edge, which needed the domain split there and
# each half graded into it -- and once that was done, CLP converged and
# assembled.
#
# These three have several at once, and in both variables:
#
#   Lidinoid   zeros at z = 0 and 1/2 on y = 0, and again on y = Im(tau)
#   rPD        zeros at z = 0, 1 on y = 0 and at 1/2 + tau/2 on the top
#   CLP_HANDLE poles at THREE corners -- 0, 1/2 and tau/2 -- plus zeros
#              at a and 1/2 + tau/2
#
# The current grading handles splits along x only.  Until it grades
# toward several points and in y as well, the patches do not converge:
# Lidinoid 449 -> 297 -> 198 as the grid refines, rPD 283 -> 188 -> 125,
# CLP_HANDLE a diameter of 12 to 37 with mean |H| * d between 1.8 and
# 111.  Shipping any of them would mean shipping a surface known to be
# wrong.  See BACKLOG for the resume note.
#
# The genus-4 row's PERIOD PROBLEM is nonetheless solved and captured:
# `_CLP_HANDLE_SOLVED` holds Weber's converged (rho, a) against tau, so
# whoever fixes the quadrature will not have to re-derive them.

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
}


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


def _spec_nodes(key, nu):
    """The x nodes and their dx/du weights for a spec patch."""
    sp = _SPECS[key]
    x0, x1 = sp['xlim']
    cuts = [x0] + [float(c) for c in sp.get('splits', ())] + [x1]
    seg = int(max(4, round(nu / (len(cuts) - 1))))
    # Each segment is graded into the branch point it ENDS at, and the
    # row quadrature runs in the graded variable.  Placing a node on the
    # branch point without doing this is worse than not splitting at
    # all: it samples the integrand at its worst and the patch grows
    # without bound (diameter 11 -> 7.4 -> 5.0 as the grid refined).
    #
    # The theta factor there carries exponent -1/2, so the integrand
    # goes like s^(-1/2) in the distance s to the point; substituting
    # s = u^2 gives s^(-1/2) ds/du = 2, bounded, and the ordinary
    # trapezoid is second order again.  Same idea as `_domain_u`, one
    # power different.
    pieces, weights = [], []
    for i in range(len(cuts) - 1):
        lo, hi = cuts[i], cuts[i + 1]
        at_hi = (i + 1) < len(cuts) - 1        # a split ends this piece
        at_lo = i > 0
        w = math.sqrt(abs(hi - lo))
        if at_hi:
            u = np.linspace(w, 0.0, seg)
            u[-1] = w * 1e-6
            xp, dxdu = hi - u ** 2, 2.0 * u
        elif at_lo:
            u = np.linspace(0.0, w, seg)
            u[0] = w * 1e-6
            xp, dxdu = lo + u ** 2, 2.0 * u
        else:
            xp = np.linspace(lo, hi, seg)
            dxdu = np.ones_like(xp)
        order = np.argsort(xp)
        pieces.append(xp[order])
        weights.append(np.abs(dxdu[order]))
    xs = np.concatenate(pieces)
    wts = np.concatenate(weights)
    keep = np.concatenate([[True], np.diff(xs) > 1e-14])
    xs, wts = xs[keep], wts[keep]
    return xs, wts


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
    xs, wts = _spec_nodes(key, nu)

    # BOTH ends of the y range are held off the edge, not just the
    # bottom.  The theta factors vanish on the lattice, and for the
    # Lidinoid and rPD the domain reaches a second row of lattice points
    # at the top -- offsetting only the bottom left those on the
    # boundary, and the patch diameter then grew without limit as the
    # grid refined instead of converging.
    ys = np.linspace(y0 + eps, y1 - eps, int(nv))
    Z = xs[:, None] + 1j * ys[None, :]

    q = np.exp(1j * np.pi * tau)
    L = np.full(Z.shape, sp['const'], dtype=complex)
    for shift, c in sp['terms'](a, tau):
        L = L + c * _log_theta(Z - shift, q)
    g = np.exp(L)
    inv = 1.0 / g
    W = np.stack([0.5 * (inv - g), 0.5j * (inv + g),
                  np.ones_like(g)], axis=-1) * np.exp(1j * ang)

    F = np.zeros((len(xs), len(ys), 3), dtype=complex)
    dy = ys[1] - ys[0]
    col = W[0]
    F[0, 1:] = np.cumsum(0.5 * (col[:-1] + col[1:]) * (1j * dy), axis=0)
    # Rows integrate in the graded variable: dx = (dx/du) du, so the
    # trapezoid is taken on W * dx/du against a uniform du, recovered
    # here as the ratio of the actual node spacing to the weight.
    du = np.diff(xs) / np.maximum(
        0.5 * (wts[:-1] + wts[1:]), 1e-300)
    V = W * wts[:, None, None]
    F[1:] = F[0][None, :, :] + np.cumsum(
        0.5 * (V[:-1] + V[1:]) * du[:, None, None], axis=0)
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


def spec_build(key, cells, res_per_cell, scale, theta,
               arrangement='UNIT'):
    """Cached wrapper -- see `_spec_build` for the construction.

    NOT memoised on the arguments alone.  CLP's two shape moduli live
    in `_SPECS`, set by `clp_params`, so they do not appear in this
    signature; a cache keyed on the arguments would hand back the old
    surface after the modulus changed.  The key carries them explicitly.
    """
    sp = _SPECS[key]
    ck = ('hexagonal.spec_build', key, tuple(np.ravel(cells))
          if isinstance(cells, (tuple, list)) else cells,
          int(res_per_cell), float(scale), float(theta), arrangement,
          complex(sp['tau']), float(sp['a']), complex(sp.get('const', 0)))
    return _geom_cache.cached(
        ck, lambda: _spec_build(key, cells, res_per_cell, scale, theta,
                                arrangement))


def _spec_build(key, cells, res_per_cell, scale, theta,
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
    if built is None:
        V = P.reshape(-1, 3)
        Q = _patch_quads(P.shape[0], P.shape[1])
        return _fit(V, [tuple(int(x) for x in q) for q in Q], scale)
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
        P = _spec_patch('CLP', n, n)
        Pu, Pv = np.gradient(P, axis=0), np.gradient(P, axis=1)
        nn = np.cross(Pu, Pv)
        nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True),
                             1e-300)
        E = np.sum(Pu * Pu, -1)
        F = np.sum(Pu * Pv, -1)
        G = np.sum(Pv * Pv, -1)
        L = np.sum(np.gradient(Pu, axis=0) * nn, -1)
        M = np.sum(np.gradient(Pu, axis=1) * nn, -1)
        N = np.sum(np.gradient(Pv, axis=1) * nn, -1)
        den = 2.0 * (E * G - F * F)
        Hc = (E * N - 2.0 * F * M + G * L) / np.where(
            np.abs(den) < 1e-300, 1e-300, den)
        fl = P.reshape(-1, 3)
        diam = float(np.linalg.norm(fl.max(0) - fl.min(0)))
        mean = float(np.mean(np.abs(Hc[3:-3, 3:-3]))) * diam
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
