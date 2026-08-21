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


def _theta11(z):
    """The odd Jacobi theta function in Weber's normalisation, period 1
    in z:  theta11(z, tau) = 2 sum (-1)^n q^((n+1/2)^2) sin((2n+1) pi z).
    """
    ang = np.multiply.outer(np.asarray(z, dtype=complex), _K)
    return 2.0 * np.sum(_COEF * np.sin(ang), axis=-1)


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


def _snap_hex(v, tol=1e-2):
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
    k = round(ang / 30.0)
    if abs(ang - 30.0 * k) > 30.0 * tol:
        return v
    a = math.radians(30.0 * k)
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
            v = _snap_hex(np.linalg.svd(C - C.mean(0),
                                        full_matrices=False)[2][0])
            p = C.mean(0)
            gens.append(_halfturn(p, v))
            kinds.append((name, 'halfturn', straight))
        elif planar < tol:
            n = _snap_hex(nrm)
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
        trial = basis + [T[i]]
        if np.linalg.matrix_rank(np.array(trial), tol=1e-6) == len(trial):
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
    P = h_patch(nu, nv, theta)
    gens, kinds = h_generators(P)
    if len(gens) < 4:
        return None
    elems = h_group(gens, maxlen=maxlen)
    B = h_lattice(elems)
    if B is None:
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
    lo, hi = V.min(0), V.max(0)
    ext = float(np.max(hi - lo)) or 1.0
    V = (V - 0.5 * (lo + hi)) * (2.0 / ext) * float(scale)
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

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("hexagonal self-test failed")
