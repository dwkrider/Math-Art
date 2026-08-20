# CMC / capillary surface generator: liquid bridges (the Delaunay
# surfaces of revolution -- unduloid and nodoid, with the catenoid and
# the cylinder as limits), sessile drops with a prescribed contact
# angle -- optionally under gravity, where the drop flattens into a
# puddle -- and free-boundary soap films spanning a fixed frame and a
# curved support (sphere or cylinder), all produced by genuine
# constrained area minimization rather than closed-form drawing.
#
# A liquid bridge spans two coaxial rings (pinned rims) at a prescribed
# enclosed volume: at the catenoid's own volume the relaxed surface IS
# the minimal catenoid (Lagrange pressure ~ 0), more volume gives the
# barrel-shaped unduloid, less gives the necked nodoid arc.  A sessile
# drop rests on a floor plane -- the contact line slides freely on the
# wall (a level-set constraint, solver/walls) while the wetting energy
#     E_wet = -cos(theta) * (wetted area)
# prescribes Young's contact angle theta; the wetted area is computed
# as a line integral around the contact line by the divergence theorem
# (Surface Evolver's technique), and in zero gravity the equilibrium
# drop is a spherical cap, which is how the result is validated.
#
# Zero gravity: the equilibrium satisfies the Young-Laplace law
# p = 2 sigma H with CONSTANT mean curvature H; the solver's Lagrange
# multiplier is the pressure, reported per run, and tests/bench
# measures H constancy, p vs 2H, the spherical-cap ground truth and
# the catenoid limit.
#
# Gravity enters as the hydrostatic potential E_g = rho g int z dV
# through evolve()'s ext_energy/ext_grad hooks (the z-moment and its
# analytic gradient are exact per-face divergence-theorem sums).  The
# equilibrium is then no longer CMC, but Young-Laplace with the
# hydrostatic head makes  2 sigma H + rho g z  constant over the free
# surface (= the Lagrange pressure at z = 0), which is the measured
# invariant; the control parameter is the Bond number Bo = rho g R0^2
# / sigma (R0 the zero-g cap radius), and for Bo >> 1 the drop is a
# puddle of thickness h = 2 l_c sin(theta/2), l_c = sqrt(sigma/rho g).
#
# A free-boundary film on a frictionless support surface meets that
# support ORTHOGONALLY (the natural boundary condition of the area
# functional); the film modes measure the achieved contact angle
# against 90 degrees with an O(h^2) local-quadric normal estimator.
#
# Pure-math core (no bpy): mesh builders, volume bookkeeping for open
# surfaces, wetting energy/gradient, and the measurement instruments
# (sphere fit, signed mean curvature) -- all reused by tests/bench.
#
# References:
# - C. E. Delaunay, "Sur la surface de revolution dont la courbure
#   moyenne est constante", Journal de Mathematiques Pures et
#   Appliquees 6 (1841), 309-314 -- the classification of CMC
#   surfaces of revolution (unduloid, nodoid, catenoid, cylinder,
#   sphere).
# - T. Young, "An essay on the cohesion of fluids", Philosophical
#   Transactions of the Royal Society 95 (1805), 65-87 -- the
#   contact-angle condition.
# - P. S. de Laplace, "Traite de mecanique celeste", Supplement au
#   livre X (1806) -- the Young-Laplace pressure law.
# - J. Plateau, "Statique experimentale et theorique des liquides
#   soumis aux seules forces moleculaires" (1873).
# - R. Finn, "Equilibrium Capillary Surfaces", Grundlehren der
#   mathematischen Wissenschaften 284, Springer (1986) -- sessile
#   drops and capillary theory.
# - P.-G. de Gennes, F. Brochard-Wyart, D. Quere, "Capillarity and
#   Wetting Phenomena" (Springer, 2004) -- capillary length,
#   Bond number, and the gravity-flattened puddle thickness
#   h = 2 l_c sin(theta/2).
# - R. Courant, "Dirichlet's Principle, Conformal Mapping, and Minimal
#   Surfaces" (Interscience, 1950) -- free boundaries of minimal
#   surfaces meet the support surface orthogonally.
# - K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#   (1992) -- constrained evolution; wetting as a contact-line
#   integral; level-set constraint walls (cnstrnt.c).

bl_info = {
    "name": "CMC Capillary Surface",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Odds & Ends",
    "description": "Liquid bridges (Delaunay surfaces) and sessile "
                   "drops by volume-constrained evolution with wall "
                   "constraints",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    import bpy
    from bpy.props import IntProperty, FloatProperty, EnumProperty, \
        BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:
    from .solver import volume as _svol
    from .solver import walls as _swalls
except ImportError:                      # flat import outside the package
    from solver import volume as _svol   # type: ignore
    from solver import walls as _swalls  # type: ignore

try:
    from .sharp_creases import mark_sharp, boundary_edges
except ImportError:                      # flat import outside the package
    from sharp_creases import mark_sharp, boundary_edges  # type: ignore


# --------------------------------------------------------------------
# Pure-math core (no bpy)
# --------------------------------------------------------------------

def catenoid_c(h=0.5, R=1.0):
    """Neck parameter of the stable (deep-root) catenoid between
    coaxial radius-R rings at z = +-h: c cosh(h/c) = R."""
    f = lambda c: c * math.cosh(h / c) - R
    # f(R) > 0 and f -> +inf as c -> 0+; for h/R below the Goldschmidt
    # limit (~0.6627) there are two roots between.  Walk down from R to
    # bracket the first crossing -- that is the deep (stable) root.
    step = R / 256.0
    hi = R
    lo = R - step
    while f(lo) > 0.0:
        hi = lo
        lo -= step
        if lo <= step / 2.0:
            raise ValueError("no catenoid spans these rings "
                             "(h/R beyond the Goldschmidt limit)")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def catenoid_area(h=0.5, R=1.0):
    """(area, c) of the minimal catenoid r = c cosh(z/c) between the
    rings: A = 2 pi c (h + (c/2) sinh(2h/c))."""
    c = catenoid_c(h, R)
    return 2.0 * math.pi * c * (h + 0.5 * c * math.sinh(2.0 * h / c)), c


def catenoid_volume(h=0.5, R=1.0):
    """Enclosed volume pi int r^2 dz of the catenoid = (c/2) * area."""
    A, c = catenoid_area(h, R)
    return 0.5 * c * A


def polygon_area(R, n):
    """Area of the regular n-gon inscribed in a circle of radius R --
    the exact area of a discrete rim disk."""
    return 0.5 * n * R * R * math.sin(2.0 * math.pi / n)


def build_bridge_mesh(nring=48, nrow=17, h=0.5, R=1.0):
    """Cylinder lathe between rings at z = +-h: (V, T, labels, fixed).
    Outward-wound, labels (0, 1), rim rows pinned via `fixed`."""
    th = np.linspace(0.0, 2.0 * np.pi, nring, endpoint=False)
    zs = np.linspace(-h, h, nrow)
    V = np.array([[R * math.cos(t), R * math.sin(t), z]
                  for z in zs for t in th])
    T = []
    for j in range(nrow - 1):
        for i in range(nring):
            a0 = j * nring + i
            b0 = j * nring + (i + 1) % nring
            T.append([a0, b0, a0 + nring])
            T.append([b0, b0 + nring, a0 + nring])
    T = np.asarray(T, dtype=np.int64)
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:nring] = True
    fixed[-nring:] = True
    labels = np.zeros((len(T), 2), dtype=np.int64)
    labels[:, 1] = 1
    return V, T, labels, fixed


def bridge_lateral_target(v_true, R, h, nring):
    """Divergence-theorem bookkeeping for the OPEN bridge surface: the
    per-face det sum over the lateral faces alone equals the true
    enclosed volume minus the two cones over the (fixed) rim polygons,
    V_lat = V_true - 2 * A_poly * h / 3, with the origin at the axis
    midpoint.  Because the rims are pinned the difference is constant,
    so constraining V_lat constrains the true volume."""
    return float(v_true) - 2.0 * polygon_area(R, nring) * h / 3.0


def bridge_true_volume(v_lat, R, h, nring):
    """Inverse of bridge_lateral_target."""
    return float(v_lat) + 2.0 * polygon_area(R, nring) * h / 3.0


def cap_geometry(theta_rad, volume):
    """Spherical-cap drop of contact angle theta and volume V on a
    plane: dict with sphere radius R, contact radius a, apex height,
    free area, wetted area, and total energy A_free - cos(theta) *
    A_wet (sigma = 1)."""
    t = float(theta_rad)
    if not (0.0 < t < math.pi):
        raise ValueError("contact angle must be in (0, 180) degrees")
    ct = math.cos(t)
    R = (3.0 * float(volume)
         / (math.pi * (1.0 - ct) ** 2 * (2.0 + ct))) ** (1.0 / 3.0)
    a = R * math.sin(t)
    A_free = 2.0 * math.pi * R * R * (1.0 - ct)
    A_wet = math.pi * a * a
    return {"R": R, "a": a, "height": R * (1.0 - ct),
            "A_free": A_free, "A_wet": A_wet,
            "E": A_free - ct * A_wet}


def build_cap_mesh(theta_deg, R=1.0, nring=48):
    """Spherical-cap lathe (contact angle theta, sphere radius R) with
    its contact line ON the plane z = 0: (V, T, labels, rim_mask).
    Rim ring first (vertices 0..nring-1), rings up to the apex,
    outward-wound.  With the boundary in a plane through the origin
    the per-face det sum IS the enclosed drop volume (the flat bottom
    contributes zero)."""
    th = math.radians(theta_deg)
    zc = -R * math.cos(th)
    a = R * math.sin(th)
    h_t = 2.0 * math.pi * a / nring
    K = max(3, int(round(R * th / h_t)))
    V = []
    for phi in np.linspace(th, 0.0, K + 1)[:-1]:
        r = R * math.sin(phi)
        z = zc + R * math.cos(phi)
        for p in range(nring):
            ang = 2.0 * math.pi * p / nring
            V.append([r * math.cos(ang), r * math.sin(ang), z])
    apex = len(V)
    V.append([0.0, 0.0, zc + R])
    T = []
    for k in range(K - 1):
        for p in range(nring):
            p2 = (p + 1) % nring
            a0, b0 = k * nring + p, k * nring + p2
            c0, d0 = (k + 1) * nring + p, (k + 1) * nring + p2
            T.append([a0, b0, c0])
            T.append([b0, d0, c0])
    last = (K - 1) * nring
    for p in range(nring):
        T.append([last + p, last + (p + 1) % nring, apex])
    V = np.asarray(V, float)
    T = np.asarray(T, dtype=np.int64)
    labels = np.zeros((len(T), 2), dtype=np.int64)
    labels[:, 1] = 1
    if _svol.region_volumes(V, T, labels)[0] < 0.0:
        T[:, [1, 2]] = T[:, [2, 1]]      # enforce outward winding
    rim = np.zeros(len(V), dtype=bool)
    rim[:nring] = True
    return V, T, labels, rim


def rim_loop(V, nring):
    """The contact-line loop (vertices 0..nring-1 of a cap mesh),
    ordered counterclockwise seen from +z so the wetted area comes out
    positive."""
    idx = np.arange(nring)
    P = V[idx]
    a2 = 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1)
                            - np.roll(P[:, 0], -1) * P[:, 1]))
    return idx if a2 > 0.0 else idx[::-1].copy()


def wetted_area(V, loop):
    """Area of the plane region enclosed by the contact line, as the
    divergence-theorem line integral (1/2) sum (p_i x p_{i+1}) . z --
    exact for the polygonal contact line, valid for the loop in the
    z = 0 plane."""
    P = V[loop]
    Q = V[np.roll(loop, -1)]
    return 0.5 * float(np.sum(P[:, 0] * Q[:, 1] - Q[:, 0] * P[:, 1]))


def wetted_area_grad(V, loop):
    """d(wetted area)/d(vertex): (1/2)(p_next - p_prev) x z_hat at
    each contact-line vertex, zero elsewhere.  Purely in-plane."""
    g = np.zeros_like(V)
    nxt = V[np.roll(loop, -1)]
    prv = V[np.roll(loop, 1)]
    g[loop, 0] = 0.5 * (nxt[:, 1] - prv[:, 1])
    g[loop, 1] = -0.5 * (nxt[:, 0] - prv[:, 0])
    return g


def drop_energy_terms(theta_rad, loop):
    """(ext_energy, ext_grad) callables for solver.volume.evolve:
    E_wet = -cos(theta) * wetted_area, Young's wetting energy with
    sigma = 1 (Evolver's contact-line integral technique)."""
    ct = math.cos(float(theta_rad))

    def ext_e(V):
        return -ct * wetted_area(V, loop)

    def ext_g(V):
        return -ct * wetted_area_grad(V, loop)

    return ext_e, ext_g


# --------------------------------------------------------------------
# Gravity: hydrostatic potential E_g = rho g * integral z dV
# --------------------------------------------------------------------

def z_moment(V, T, labels, region=1):
    """integral of z over the region's volume, by the divergence
    theorem with F = (z^2/2) z_hat: over the region's outward-oriented
    faces, sum of Az * mean(z^2)/2 with Az the face's signed projected
    area onto the xy-plane -- exact for flat triangles (midpoint rule
    on the quadratic z^2).  For the sessile drop the missing floor
    faces lie in z = 0 and would contribute zero, so the open-cap sum
    IS the drop's z-moment."""
    labels = np.asarray(labels)
    s = ((labels[:, 1] == region).astype(float)
         - (labels[:, 0] == region))
    P0, P1, P2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    Az = 0.5 * ((P1[:, 0] - P0[:, 0]) * (P2[:, 1] - P0[:, 1])
                - (P1[:, 1] - P0[:, 1]) * (P2[:, 0] - P0[:, 0]))
    z0, z1, z2 = P0[:, 2], P1[:, 2], P2[:, 2]
    Q = (z0 * z0 + z1 * z1 + z2 * z2
         + z0 * z1 + z1 * z2 + z2 * z0) / 12.0
    return float(np.sum(s * Az * Q))


def z_moment_grad(V, T, labels, region=1):
    """Analytic d(z_moment)/d(vertex), shape (n, 3): the xy components
    differentiate the projected area Az (shoelace), the z component
    the quadratic z-mean."""
    labels = np.asarray(labels)
    s = ((labels[:, 1] == region).astype(float)
         - (labels[:, 0] == region))
    P0, P1, P2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    Az = 0.5 * ((P1[:, 0] - P0[:, 0]) * (P2[:, 1] - P0[:, 1])
                - (P1[:, 1] - P0[:, 1]) * (P2[:, 0] - P0[:, 0]))
    zs = (P0[:, 2], P1[:, 2], P2[:, 2])
    Q = (zs[0] * zs[0] + zs[1] * zs[1] + zs[2] * zs[2]
         + zs[0] * zs[1] + zs[1] * zs[2] + zs[2] * zs[0]) / 12.0
    sQ = s * Q
    sAz = s * Az
    Ps = (P0, P1, P2)
    g = np.zeros_like(V)
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        contrib = np.empty((len(T), 3))
        contrib[:, 0] = sQ * 0.5 * (Ps[b][:, 1] - Ps[c][:, 1])
        contrib[:, 1] = sQ * 0.5 * (Ps[c][:, 0] - Ps[b][:, 0])
        contrib[:, 2] = sAz * (2.0 * zs[a] + zs[b] + zs[c]) / 12.0
        np.add.at(g, T[:, a], contrib)
    return g


def gravity_energy_terms(rho_g, T, labels, region=1):
    """(ext_energy, ext_grad) callables for solver.volume.evolve:
    E_g = rho_g * integral z dV at sigma = 1 (rho_g = rho * g)."""
    rg = float(rho_g)

    def ext_e(V):
        return rg * z_moment(V, T, labels, region)

    def ext_g(V):
        return rg * z_moment_grad(V, T, labels, region)

    return ext_e, ext_g


def capillary_length(rho_g):
    """l_c = sqrt(sigma / rho g) at sigma = 1."""
    return 1.0 / math.sqrt(float(rho_g))


def puddle_height(theta_rad, rho_g):
    """Asymptotic (Bo -> inf) puddle thickness 2 l_c sin(theta/2)
    (de Gennes-Brochard-Wyart-Quere): gravity flattens a large drop to
    this uniform depth, the balance of hydrostatic spreading against
    the wetting-limited edge."""
    return 2.0 * capillary_length(rho_g) * math.sin(0.5 * float(theta_rad))


def bond_number(rho_g, theta_rad, volume):
    """Bo = rho g R0^2 / sigma with R0 the ZERO-gravity spherical-cap
    radius for this volume and contact angle -- the dimensionless
    gravity strength (Bo << 1: spherical cap; Bo >> 1: puddle)."""
    R0 = cap_geometry(theta_rad, volume)["R"]
    return float(rho_g) * R0 * R0


def rho_g_for_bond(bond, theta_rad, volume):
    """Inverse of bond_number: the rho*g giving the requested Bo."""
    R0 = cap_geometry(theta_rad, volume)["R"]
    return float(bond) / (R0 * R0)


def cap_angle_for_height(volume, height, lo=5.0, hi=175.0):
    """Contact angle (degrees) whose spherical cap of the given volume
    has apex height `height` (bisection; height is monotone in the
    angle at fixed volume)."""
    f = lambda th: cap_geometry(math.radians(th), volume)["height"] \
        - float(height)
    if f(lo) > 0.0:
        return lo
    if f(hi) < 0.0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sphere_fit(P):
    """Algebraic least-squares sphere: (center, radius, rms residual)."""
    P = np.asarray(P, float)
    A = np.concatenate([2.0 * P, np.ones((len(P), 1))], axis=1)
    b = np.einsum('ij,ij->i', P, P)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(max(sol[3] + float(c @ c), 0.0))
    res = float(np.sqrt(np.mean(
        (np.linalg.norm(P - c, axis=1) - r) ** 2)))
    return c, r, res


def signed_mean_curvature(V, T):
    """Signed discrete mean curvature per vertex: raw cotan Laplacian
    over twice the barycentric vertex area, signed against the
    area-weighted vertex normal (positive where the surface curves
    away from its outward normal, e.g. +1/r on an outward-wound
    sphere).  Boundary and pinned vertices carry no meaning here --
    mask them out before taking statistics."""
    n = len(V)
    lap = np.zeros((n, 3))
    Astar = np.zeros(n)
    nrm = np.zeros((n, 3))
    for (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = V[T[:, a]] - V[T[:, c]]
        w = V[T[:, b]] - V[T[:, c]]
        cr = np.cross(u, w)
        crn = np.maximum(np.linalg.norm(cr, axis=1), 1e-300)
        cot = np.einsum('ij,ij->i', u, w) / crn
        d = V[T[:, b]] - V[T[:, a]]
        contrib = 0.5 * cot[:, None] * d
        np.add.at(lap, T[:, a], contrib)
        np.add.at(lap, T[:, b], -contrib)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    tri_area = 0.5 * np.linalg.norm(fn, axis=1)
    for k in range(3):
        np.add.at(Astar, T[:, k], tri_area / 3.0)
        np.add.at(nrm, T[:, k], fn)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
    return -np.einsum('ij,ij->i', lap, nrm) \
        / (2.0 * np.maximum(Astar, 1e-300))


def local_contact_angles(V, T, loop):
    """Per-vertex contact angle (degrees) of a drop against the floor
    z = 0, measured LOCALLY at the contact line from O(h^2) quadric
    surface normals: theta = atan2(-m.z, m.e_out) with m the film's
    outward boundary conormal and e_out the outward horizontal radial.
    Unlike the global sphere fit this needs no spherical shape, so it
    works for gravity-flattened puddles -- Young's angle is a local
    force balance and must hold at any Bond number."""
    V = np.asarray(V, float)
    loop = np.asarray(loop, dtype=np.int64)
    tv = V[np.roll(loop, -1)] - V[np.roll(loop, 1)]
    tv /= np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-300)
    nf = quadric_vertex_normals(V, T, loop)
    m = np.cross(nf, tv)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-300)
    e_out = V[loop].copy()
    e_out[:, 2] = 0.0
    e_out /= np.maximum(np.linalg.norm(e_out, axis=1, keepdims=True),
                        1e-300)
    # orient m away from the film: the surface interior must lie on
    # the -m side (test against the mean of each rim vertex's
    # off-loop neighbours; works for wetting AND beading angles,
    # where the radial sign of m legitimately differs)
    nbr = _vertex_neighbors(T, len(V))
    on_loop = np.zeros(len(V), dtype=bool)
    on_loop[loop] = True
    for row, v in enumerate(loop):
        ins = [u for u in nbr[v] if not on_loop[u]]
        if ins:
            d = np.mean(V[ins], axis=0) - V[v]
            if float(m[row] @ d) > 0.0:
                m[row] *= -1.0
    return np.degrees(np.arctan2(-m[:, 2],
                                 np.einsum('ij,ij->i', m, e_out)))


def measured_contact_angle(V, T):
    """Contact angle (degrees) of a relaxed drop read from the sphere
    fitted to its free surface: cos(theta) = -z_center / R_fit (the
    floor is z = 0).  Also returns the fit rms."""
    ids = np.unique(np.asarray(T).ravel())
    c, r, rms = sphere_fit(np.asarray(V, float)[ids])
    cosv = max(-1.0, min(1.0, -float(c[2]) / max(r, 1e-300)))
    return math.degrees(math.acos(cosv)), rms


def relax_bridge(volume_factor=1.15, nring=48, nrow=17, h=0.5, R=1.0,
                 iters=600, groom_every=0, squash=0.0, optimizer=None):
    """Build and evolve a liquid bridge at `volume_factor` x the
    cylinder volume (factor catenoid_volume/V_cyl ~ 0.809 for h/R=0.5
    gives the minimal catenoid; more is an unduloid barrel, less a
    nodoid neck).  `squash` bulges the free rows of the seed radially
    by 1 + squash * cos(pi z / 2h), so the evolution demonstrably
    recovers the equilibrium rather than starting on it.
    Returns (V, T, info)."""
    V, T, labels, fixed = build_bridge_mesh(nring, nrow, h, R)
    v_lat0 = _svol.region_volumes(V, T, labels)[0]
    v_true0 = bridge_true_volume(v_lat0, R, h, nring)
    target = bridge_lateral_target(volume_factor * v_true0, R, h, nring)
    if squash:
        fac = 1.0 + float(squash) * np.cos(
            np.pi * V[:, 2] / (2.0 * h))
        fac[fixed] = 1.0
        V[:, 0] *= fac
        V[:, 1] *= fac
    info = _svol.evolve(V, T, labels, targets=[target], iters=iters,
                        fixed=fixed, groom_every=groom_every,
                        optimizer=optimizer)
    return V, T, info


def relax_drop(theta_deg=60.0, volume=2.0 * math.pi / 3.0, nring=48,
               iters=1000, groom_every=4, seed_theta_deg=90.0,
               rho_g=0.0, optimizer=None):
    """Build and evolve a sessile drop: contact line sliding on the
    floor plane (two-sided wall on the rim, one-sided on the interior),
    wetting energy -cos(theta) * wetted area, volume constrained.
    The seed is a spherical cap of angle seed_theta_deg at the target
    volume, so the run demonstrably MOVES the contact line to the
    prescribed angle.  rho_g > 0 adds the hydrostatic energy
    rho g * int z dV (sigma = 1): the drop flattens toward the puddle
    of thickness 2 l_c sin(theta/2); with strong gravity the seed cap
    is automatically squashed toward that height so the run does not
    have to collapse a tall cap.  rho_g = 0 is byte-identical to the
    original zero-gravity path.  Returns (V, T, info)."""
    if rho_g:
        h_inf = puddle_height(math.radians(theta_deg), rho_g)
        nat = cap_geometry(math.radians(seed_theta_deg), volume)
        if nat["height"] > 1.5 * h_inf:
            seed_theta_deg = cap_angle_for_height(volume, 1.2 * h_inf)
    geo_seed = cap_geometry(math.radians(seed_theta_deg), volume)
    V, T, labels, rim = build_cap_mesh(seed_theta_deg, geo_seed["R"],
                                       nring)
    interior = ~rim
    loop = rim_loop(V, nring)
    ext_e, ext_g = drop_energy_terms(math.radians(theta_deg), loop)
    if rho_g:
        ge, gg = gravity_energy_terms(rho_g, T, labels)
        ew, gw = ext_e, ext_g
        ext_e = lambda V: ew(V) + ge(V)          # noqa: E731
        ext_g = lambda V: gw(V) + gg(V)          # noqa: E731
    plane = _swalls.PlaneWall([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    floor = _swalls.PlaneWall([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                              one_sided=True)
    info = _svol.evolve(V, T, labels, targets=[volume], iters=iters,
                        groom_every=groom_every,
                        walls=[(plane, rim), (floor, interior)],
                        ext_energy=ext_e, ext_grad=ext_g,
                        optimizer=optimizer)
    return V, T, info


# --------------------------------------------------------------------
# Free-boundary soap films on curved supports (sphere / cylinder)
# --------------------------------------------------------------------

def build_film_strip(outer, inner, nrow):
    """Welded annular strip between two closed loops of equal length:
    row 0 = outer (the fixed frame), row nrow-1 = inner (the free
    boundary that will slide on the support wall), rows in between
    linearly interpolated.  Returns (V, T, labels, fixed, freeb) with
    labels all (0, 0): an OPEN film enclosing no body, so evolve()
    runs pure area descent (the Gram system is 0 x 0)."""
    outer = np.asarray(outer, float)
    inner = np.asarray(inner, float)
    nphi = len(outer)
    ts = np.linspace(0.0, 1.0, int(nrow))
    V = np.concatenate([(1.0 - t) * outer + t * inner for t in ts])
    T = []
    for j in range(int(nrow) - 1):
        for i in range(nphi):
            a0 = j * nphi + i
            b0 = j * nphi + (i + 1) % nphi
            T.append([a0, b0, a0 + nphi])
            T.append([b0, b0 + nphi, a0 + nphi])
    T = np.asarray(T, dtype=np.int64)
    labels = np.zeros((len(T), 2), dtype=np.int64)
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:nphi] = True
    freeb = np.zeros(len(V), dtype=bool)
    freeb[-nphi:] = True
    return V, T, labels, fixed, freeb


def film_sphere_seed(R_support=0.8, ring_r=1.6, ring_z=0.0, nphi=48,
                     nrow=None, seed_shift=0.0):
    """Film spanning from a fixed horizontal ring (radius ring_r at
    height ring_z) to a sphere of radius R_support at the origin.  The
    free inner loop is seeded at the radial projection of the ring
    onto the sphere, optionally biased by seed_shift along z BEFORE
    normalizing -- so the contact line starts demonstrably off its
    equilibrium and must slide there.  Returns
    (V, T, labels, fixed, freeb, wall)."""
    th = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    outer = np.stack([ring_r * np.cos(th), ring_r * np.sin(th),
                      np.full(nphi, float(ring_z))], axis=1)
    dirs = outer.copy()
    dirs[:, 2] += float(seed_shift)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    inner = float(R_support) * dirs
    if nrow is None:
        gap = float(np.mean(np.linalg.norm(outer - inner, axis=1)))
        h = 2.0 * math.pi * ring_r / nphi
        nrow = max(4, int(round(gap / h)) + 1)
    V, T, labels, fixed, freeb = build_film_strip(outer, inner, nrow)
    wall = _swalls.SphereWall([0.0, 0.0, 0.0], R_support)
    return V, T, labels, fixed, freeb, wall


def film_cylinder_seed(R_support=0.5, ring_r=1.3, tilt_deg=35.0,
                       nphi=48, nrow=None):
    """Film spanning from a fixed TILTED ring (radius ring_r, centered
    on the axis at z = 0, tilted about x by tilt_deg) inward to a
    vertical cylindrical column of radius R_support about the z axis
    -- a sail on a mast.  The free boundary is seeded at the ring's
    azimuthal projection onto the column.  Returns
    (V, T, labels, fixed, freeb, wall)."""
    a = math.radians(float(tilt_deg))
    th = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    outer = np.stack([ring_r * np.cos(th),
                      ring_r * np.sin(th) * math.cos(a),
                      ring_r * np.sin(th) * math.sin(a)], axis=1)
    psi = np.arctan2(outer[:, 1], outer[:, 0])
    inner = np.stack([R_support * np.cos(psi), R_support * np.sin(psi),
                      outer[:, 2]], axis=1)
    if nrow is None:
        gap = float(np.mean(np.linalg.norm(outer - inner, axis=1)))
        h = 2.0 * math.pi * ring_r / nphi
        nrow = max(4, int(round(gap / h)) + 1)
    V, T, labels, fixed, freeb = build_film_strip(outer, inner, nrow)
    wall = _swalls.CylinderWall([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                R_support)
    return V, T, labels, fixed, freeb, wall


def film_disk_seed(R_support=1.0, nphi=48, tilt=0.25, bulge=0.35):
    """Disk film spanning INSIDE a cylindrical tube of radius
    R_support about the z axis: the ENTIRE boundary is free on the
    wall (no fixed frame at all).  Seeded tilted (boundary
    z = tilt cos(phi)) and bulged, so the relaxation must flatten it
    to the perpendicular cross-section disk of area pi R^2.  Returns
    (V, T, labels, fixed, freeb, wall)."""
    K = max(3, nphi // 4)
    V = []
    for k in range(K):
        t = k / K
        r = R_support * (1.0 - t)
        for p in range(nphi):
            phi = 2.0 * math.pi * p / nphi
            V.append([r * math.cos(phi), r * math.sin(phi),
                      (1.0 - t) * tilt * math.cos(phi) + bulge * t])
    apex = len(V)
    V.append([0.0, 0.0, float(bulge)])
    T = []
    for k in range(K - 1):
        for p in range(nphi):
            p2 = (p + 1) % nphi
            a0, b0 = k * nphi + p, k * nphi + p2
            c0, d0 = (k + 1) * nphi + p, (k + 1) * nphi + p2
            T.append([a0, b0, c0])
            T.append([b0, d0, c0])
    last = (K - 1) * nphi
    for p in range(nphi):
        T.append([last + p, last + (p + 1) % nphi, apex])
    V = np.asarray(V, float)
    T = np.asarray(T, dtype=np.int64)
    labels = np.zeros((len(T), 2), dtype=np.int64)
    fixed = np.zeros(len(V), dtype=bool)
    freeb = np.zeros(len(V), dtype=bool)
    freeb[:nphi] = True
    wall = _swalls.CylinderWall([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                R_support)
    return V, T, labels, fixed, freeb, wall


def redistribute_boundary(V, T, wall, freeb, lam=0.5):
    """Even out the free-boundary vertex spacing IN PLACE: move each
    boundary vertex toward its loop-neighbour midpoint, keeping only
    the component ALONG the local boundary tangent (so the boundary
    curve itself is preserved to O(h^2)), then Newton-project back
    onto the wall.  Needed because sliding along the boundary is
    area-neutral -- the descent leaves the spacing underdetermined and
    disorder accumulates until the boundary strip degenerates (the
    same tangential-disorder mechanism the wall-constraints plan
    documented for interior vertices, which grooming fixes everywhere
    EXCEPT on the pinned non-manifold/boundary vertices)."""
    loop = boundary_loop(T, freeb)
    if len(loop) < 3:
        return
    P = V[loop]
    t = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
    d = 0.5 * (np.roll(P, -1, axis=0) + np.roll(P, 1, axis=0)) - P
    V[loop] += lam * np.einsum('ij,ij->i', d, t)[:, None] * t
    _swalls.newton_project(wall, V, freeb)


def relax_film(V, T, labels, fixed, wall, freeb, iters=600,
               groom_every=4):
    """Relax an open film (labels all (0,0), no volume constraint):
    pure area descent with the free-boundary vertices sliding on the
    support wall, and (at groom cadence) the boundary spacing
    redistributed along the contact curve.  Returns the evolve info
    dict (pressures empty).

    Films deliberately have NO optimizer knob: a free-boundary film
    whose contact loop winds its support keeps its winding class only
    through an energy barrier, and L-BFGS was MEASURED to slide
    boundary vertices past each other and unwind the disk-in-cylinder
    film monotonically (area 3.13 -> 1e-4 through non-embedded
    states, winding 1 -> 0.125; a 0.5x step cap still leaves folds
    that put the area below the pi Douglas bound), while CG preserves
    the winding even at 4000 iterations -- and on the anchored
    sphere/column films L-BFGS showed no quality-or-time win to
    justify the risk."""
    hook = (None if not groom_every
            else lambda Vv, Tt: redistribute_boundary(Vv, Tt, wall,
                                                      freeb))
    return _svol.evolve(V, T, labels, iters=iters, fixed=fixed,
                        groom_every=groom_every,
                        walls=[(wall, freeb)], groom_hook=hook)


def boundary_loop(T, mask):
    """Ordered vertex loop of the open mesh boundary restricted to the
    masked vertices (edges used by exactly one face).  Robust to
    grooming: derived from the CURRENT triangulation."""
    de = np.sort(np.concatenate(
        [T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(de, axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    mask = np.asarray(mask, bool)
    bnd = bnd[mask[bnd[:, 0]] & mask[bnd[:, 1]]]
    if len(bnd) == 0:
        return np.zeros(0, dtype=np.int64)
    nxt = {}
    for a, b in bnd:
        nxt.setdefault(int(a), []).append(int(b))
        nxt.setdefault(int(b), []).append(int(a))
    start = int(bnd[0, 0])
    loop = [start]
    prev = -1
    while True:
        cands = [c for c in nxt[loop[-1]] if c != prev]
        if not cands:
            break
        prev = loop[-1]
        loop.append(cands[0])
        if loop[-1] == start:
            loop.pop()
            break
    return np.asarray(loop, dtype=np.int64)


def _vertex_neighbors(T, n):
    nbr = [set() for _ in range(n)]
    for a, b, c in T:
        nbr[a].update((b, c))
        nbr[b].update((a, c))
        nbr[c].update((a, b))
    return nbr


def quadric_vertex_normals(V, T, idx):
    """O(h^2) surface normals at the requested vertices: local frame
    from the area-weighted vertex normal, least-squares quadric height
    fit h(u,w) = c0 + c1 u + c2 w + c3 u^2 + c4 uw + c5 w^2 over the
    2-ring, normal from the fitted gradient at the vertex.  This is
    the standard cure for the O(h) secant tilt of raw face normals
    (the same reason the bubble bench fits films before measuring
    Plateau angles)."""
    V = np.asarray(V, float)
    n = len(V)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    vn = np.zeros((n, 3))
    for k in range(3):
        np.add.at(vn, T[:, k], fn)
    nbr = _vertex_neighbors(T, n)
    out = np.zeros((len(idx), 3))
    for row, v in enumerate(idx):
        ring = set(nbr[v])
        for u in list(ring):
            ring |= nbr[u]
        ring.discard(v)
        P = V[sorted(ring)] - V[v]
        e3 = vn[v] / max(np.linalg.norm(vn[v]), 1e-300)
        e1 = np.cross(e3, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(e3, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(e3, e1)
        u = P @ e1
        w = P @ e2
        h = P @ e3
        A = np.stack([np.ones_like(u), u, w, u * u, u * w, w * w],
                     axis=1)
        c, *_ = np.linalg.lstsq(A, h, rcond=None)
        nl = -c[1] * e1 - c[2] * e2 + e3
        out[row] = nl / np.linalg.norm(nl)
    return out


def film_contact_angle_dev(V, T, wall, loop):
    """Deviation (degrees, per free-boundary vertex) of the film-
    support contact angle from the free-boundary condition's 90:
    the film's boundary conormal m (surface normal x boundary tangent)
    must be PARALLEL to the wall normal for orthogonal contact, so the
    deviation is the angle between m and the wall-normal axis.
    Returns (dev_fit, dev_raw): surface normals from the O(h^2)
    quadric fit and from averaged raw incident-face normals (O(h))."""
    V = np.asarray(V, float)
    loop = np.asarray(loop, dtype=np.int64)
    tv = V[np.roll(loop, -1)] - V[np.roll(loop, 1)]
    tv /= np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-300)
    nw = wall.grad(V[loop])
    nw /= np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-300)
    nfit = quadric_vertex_normals(V, T, loop)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, T[:, k], fn)
    nraw = vn[loop]
    nraw /= np.maximum(np.linalg.norm(nraw, axis=1, keepdims=True),
                       1e-300)
    out = []
    for nf in (nfit, nraw):
        m = np.cross(nf, tv)
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True),
                        1e-300)
        c = np.abs(np.einsum('ij,ij->i', m, nw))
        out.append(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))
    return out[0], out[1]


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

if _IN_BLENDER:

    class MESH_OT_cmc_capillary_add(bpy.types.Operator):
        """Capillary surface by constrained evolution: a liquid bridge
        between two rings (Delaunay's unduloid/nodoid family, catenoid
        at the right volume), a sessile drop with a prescribed contact
        angle -- optionally flattened by gravity into a puddle -- or a
        free-boundary soap film sliding on a sphere or column"""
        bl_idname = "mesh.cmc_capillary_add"
        bl_label = "CMC Capillary Surface"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Surface",
            items=[('BRIDGE', "Liquid Bridge",
                    "Surface spanning two coaxial rings at fixed "
                    "enclosed volume: Delaunay's unduloid (fat) / "
                    "nodoid (thin) family, the catenoid at the "
                    "volume of the minimal surface"),
                   ('DROP', "Sessile Drop",
                    "Drop resting on a floor at fixed volume, the "
                    "contact line sliding to Young's contact angle; "
                    "with gravity (Bond number > 0) it flattens "
                    "toward a puddle of depth 2 l_c sin(theta/2)"),
                   ('FILM_SPHERE', "Film on Sphere",
                    "Soap film spanning a fixed ring and a sphere, "
                    "the free boundary sliding on the sphere to meet "
                    "it at 90 degrees"),
                   ('FILM_COLUMN', "Film on Column",
                    "Soap film spanning a fixed tilted ring and a "
                    "vertical column, the free boundary sliding on "
                    "the column to meet it at 90 degrees")],
            default='BRIDGE')
        volume_factor: FloatProperty(
            name="Volume Factor", default=1.15, min=0.55, max=1.8,
            description="Bridge volume as a fraction of the cylinder "
                        "volume: 1 = cylinder, ~0.809 = catenoid "
                        "(at the default aspect), more = unduloid "
                        "barrel, less = nodoid neck")
        aspect: FloatProperty(
            name="Half-Height / Radius", default=0.5, min=0.15,
            max=0.66,
            description="Ring half-separation over ring radius "
                        "(0.6627 is the Goldschmidt limit beyond "
                        "which no catenoid spans the rings)")
        contact_angle: FloatProperty(
            name="Contact Angle", default=60.0, min=15.0, max=165.0,
            subtype='UNSIGNED',
            description="Young contact angle in degrees (< 90 wets "
                        "the floor, > 90 beads up)")
        bond: FloatProperty(
            name="Bond Number", default=0.0, min=0.0, max=60.0,
            description="Gravity strength Bo = rho g R^2 / sigma "
                        "(R the zero-gravity drop radius): 0 = "
                        "spherical cap, >> 1 = flattened puddle of "
                        "depth 2 l_c sin(theta/2)")
        support_radius: FloatProperty(
            name="Support Radius", default=0.55, min=0.15, max=0.9,
            description="Radius of the supporting sphere / column, "
                        "as a fraction of the frame ring radius")
        ring_height: FloatProperty(
            name="Ring Height", default=0.45, min=-0.85, max=0.85,
            description="Height of the frame ring above the sphere "
                        "center (fraction of the ring radius); 0 "
                        "gives the flat equatorial film")
        ring_tilt: FloatProperty(
            name="Ring Tilt", default=35.0, min=0.0, max=60.0,
            description="Tilt of the frame ring around the column, "
                        "in degrees (0 gives the flat annulus)")
        show_support: BoolProperty(
            name="Support Surface", default=True,
            description="Also emit the supporting sphere / column "
                        "surface into the mesh")
        resolution: IntProperty(
            name="Resolution", default=48, min=16, max=128,
            description="Vertices around the rim / contact line")
        iterations: IntProperty(
            name="Evolve Iterations", default=800, min=0, max=5000,
            description="Constrained area-descent iterations "
                        "(0 shows the raw seed)")
        optimizer: EnumProperty(
            name="Optimizer",
            items=[('CG', "Conjugate Gradient",
                    "The established projected Polak-Ribiere "
                    "descent with the Evolver line search"),
                   ('LBFGS', "L-BFGS (Laplacian-seeded)",
                    "Quasi-Newton descent seeded by a cotan-"
                    "Laplacian solve: drops reach their contact "
                    "angle in far fewer iterations (measured: the "
                    "135-degree drop converges ~17x faster in wall "
                    "time); on bridges the vertex distribution can "
                    "drift tangentially at equilibrium, so CG "
                    "remains the recommended bridge setting.  Film "
                    "modes always use CG (measured winding-safety "
                    "hazard)")],
            default='CG',
            description="Outer descent algorithm for the "
                        "constrained evolution (BRIDGE and DROP "
                        "modes; films always use CG)")
        smooth: BoolProperty(name="Smooth Shading", default=True)
        sharp_edges: BoolProperty(
            name="Sharp Rims", default=True,
            description="Crease the surface's boundary rims -- the "
                        "bridge's pinned rings, the drop's contact "
                        "line, the film's frame ring and free "
                        "contact line -- so a Subdivision Surface "
                        "keeps them pinned to the rings / floor / "
                        "wall instead of shrinking the open boundary")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def _optimizer(self):
            return "lbfgs" if self.optimizer == 'LBFGS' else None

        def execute(self, context):
            support = None
            if self.mode == 'BRIDGE':
                nrow = max(5, 2 * int(round(
                    self.aspect * self.resolution / (2.0 * math.pi)))
                    * 2 + 1)
                V, T, info = relax_bridge(
                    volume_factor=self.volume_factor,
                    nring=self.resolution, nrow=nrow,
                    h=self.aspect, R=1.0,
                    iters=self.iterations,
                    optimizer=self._optimizer())
                name = "Liquid Bridge"
                Hs = signed_mean_curvature(V, T)
                # interior rows only (rims are pinned)
                nring = self.resolution
                Hmean = float(np.mean(Hs[nring:-nring])) \
                    if len(V) > 2 * nring else 0.0
                extra = f"H={Hmean:.4f}"
            elif self.mode == 'DROP':
                vol = 2.0 * math.pi / 3.0
                rho_g = (rho_g_for_bond(
                    self.bond, math.radians(self.contact_angle), vol)
                    if self.bond > 0.0 else 0.0)
                V, T, info = relax_drop(
                    theta_deg=self.contact_angle,
                    nring=self.resolution,
                    iters=self.iterations, rho_g=rho_g,
                    optimizer=self._optimizer())
                name = "Sessile Drop"
                if rho_g:
                    h_inf = puddle_height(
                        math.radians(self.contact_angle), rho_g)
                    extra = (f"Bo={self.bond:.1f}, apex "
                             f"{float(V[:, 2].max()):.3f} (puddle "
                             f"limit {h_inf:.3f})")
                else:
                    ang, _rms = measured_contact_angle(V, T)
                    extra = f"contact angle {ang:.2f} deg " \
                            f"(asked {self.contact_angle:.1f})"
            else:
                if self.mode == 'FILM_SPHERE':
                    V, T, labels, fixed, freeb, wall = \
                        film_sphere_seed(
                            R_support=self.support_radius,
                            ring_r=1.0, ring_z=self.ring_height,
                            nphi=self.resolution)
                    name = "Film on Sphere"
                else:
                    V, T, labels, fixed, freeb, wall = \
                        film_cylinder_seed(
                            R_support=self.support_radius,
                            ring_r=1.0, tilt_deg=self.ring_tilt,
                            nphi=self.resolution)
                    name = "Film on Column"
                info = relax_film(V, T, labels, fixed, wall, freeb,
                                  iters=self.iterations)
                loop = boundary_loop(T, freeb)
                dev_fit, _dev_raw = film_contact_angle_dev(
                    V, T, wall, loop)
                rms = float(np.sqrt(np.mean(dev_fit ** 2)))
                extra = f"contact angle 90 deg +- {rms:.2f} rms"
                if self.show_support:
                    support = self._support_mesh(V)

            # every mode's physical crease lines -- pinned rims, the
            # drop's contact line, the film's frame ring and free
            # contact line -- are the FILM mesh's open boundary, so
            # collect them before any support geometry is appended
            # (the support prop's own truncation rims are cuts, not
            # physics, and stay uncreased)
            bedges = boundary_edges(T) if self.sharp_edges else []

            if support is not None:
                SV, SF = support
                base = len(V)
                V = np.concatenate([V, SV])
                T = list(map(list, T)) + [[base + i for i in f]
                                          for f in SF]
            lo = V.min(axis=0)
            hi = V.max(axis=0)
            ctr = 0.5 * (lo + hi)
            half = float(np.max(hi - lo)) / 2.0 or 1.0
            s = self.scale / half
            Vt = (V - ctr) * s

            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(p) for p in Vt], [],
                           [list(t) for t in T])
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [self.smooth] * len(me.polygons))
            # boundary edges have a single face, so sharp shading has
            # nothing to split there; the crease weight is what
            # matters (it pins the rims under Subdivision Surface)
            ncrease = mark_sharp(me, bedges)
            if self.sharp_edges and ncrease == 0:
                # all four modes are open surfaces: a zero count
                # means the crease bookkeeping silently broke
                self.report({'WARNING'},
                            "no boundary edges found to crease")
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            ptxt = (f"pressure {float(info['pressures'][0]):.4f}, "
                    if len(info["pressures"]) else "")
            self.report({'INFO'},
                        f"{name}: {info['iters_run']} iterations, "
                        f"area {info['area']:.4f}, "
                        f"{ncrease} rim crease edges, {ptxt}{extra}")
            return {'FINISHED'}

        def _support_mesh(self, filmV):
            """Mesh of the supporting surface: an icosphere, or a
            column tube spanning a margin past the film's z range."""
            R = float(self.support_radius)
            if self.mode == 'FILM_SPHERE':
                try:
                    from .surfaces.primitives import icosphere
                except ImportError:
                    from surfaces.primitives import icosphere  # type: ignore
                SV, SF = icosphere(3, 'per_level')
                return R * np.asarray(SV, float), \
                    [list(f) for f in np.asarray(SF)]
            zlo = float(filmV[:, 2].min()) - 0.35
            zhi = float(filmV[:, 2].max()) + 0.35
            n = int(self.resolution)
            th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            ring = np.stack([R * np.cos(th), R * np.sin(th),
                             np.zeros(n)], axis=1)
            SV = np.concatenate([ring + [0.0, 0.0, zlo],
                                 ring + [0.0, 0.0, zhi]])
            SF = []
            for i in range(n):
                j = (i + 1) % n
                SF.append([i, j, n + j, n + i])
            return SV, SF

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'BRIDGE':
                lay.prop(self, 'volume_factor')
                lay.prop(self, 'aspect')
            elif self.mode == 'DROP':
                lay.prop(self, 'contact_angle')
                lay.prop(self, 'bond')
            else:
                lay.prop(self, 'support_radius')
                if self.mode == 'FILM_SPHERE':
                    lay.prop(self, 'ring_height')
                else:
                    lay.prop(self, 'ring_tilt')
                lay.prop(self, 'show_support')
            lay.prop(self, 'resolution')
            lay.prop(self, 'iterations')
            if self.mode in ('BRIDGE', 'DROP'):
                lay.prop(self, 'optimizer')
            for k in ('smooth', 'sharp_edges', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.cmc_capillary_add",
                             icon='MATFLUID')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_cmc_capillary_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_cmc_capillary_add)


# --------------------------------------------------------------------
# self-test (headless; the full ground-truth battery is tests/bench)
# --------------------------------------------------------------------

def _selftest():
    ok = True

    # catenoid parameters against the bench's independent bisection
    A_cat, c = catenoid_area(0.5, 1.0)
    good = abs(c * math.cosh(0.5 / c) - 1.0) < 1e-12 \
        and abs(c - 0.848338) < 1e-4
    ok &= good
    print(f"cmc: catenoid c = {c:.6f} (deep root) "
          f"{'OK' if good else 'FAIL'}")

    # bridge volume bookkeeping is EXACT on the prism: lateral det sum
    # + two rim cones = polygon-cylinder volume
    V, T, labels, fixed = build_bridge_mesh(48, 9, 0.5, 1.0)
    v_lat = _svol.region_volumes(V, T, labels)[0]
    v_true = bridge_true_volume(v_lat, 1.0, 0.5, 48)
    v_prism = polygon_area(1.0, 48) * 1.0
    good = abs(v_true - v_prism) < 1e-12
    ok &= good
    print(f"cmc: prism volume bookkeeping {v_true:.12f} vs "
          f"{v_prism:.12f} {'OK' if good else 'FAIL'}")

    # cap mesh volume converges to the analytic cap volume
    geo = cap_geometry(math.radians(75.0), 1.0)
    errs = []
    for nring in (24, 48):
        Vc, Tc, Lc, rim = build_cap_mesh(75.0, geo["R"], nring)
        errs.append(abs(_svol.region_volumes(Vc, Tc, Lc)[0] - 1.0))
    good = errs[1] < errs[0] / 3.0 and errs[1] < 8e-3
    ok &= good
    print(f"cmc: cap volume error {errs[0]:.2e} -> {errs[1]:.2e} "
          f"under refinement {'OK' if good else 'FAIL'}")

    # rim creasing: the physical crease lines of every mode are the
    # open mesh boundary the operator marks; counts must be exact and
    # NON-ZERO (an empty crease set is the silent regression mode).
    # bridge: two pinned rims; drop: the contact line (the wetted
    # disk is not meshed); films: frame ring + free contact line
    nb = len(boundary_edges(build_bridge_mesh(48, 9, 0.5, 1.0)[1]))
    nc = len(boundary_edges(Tc))
    nf = len(boundary_edges(film_sphere_seed(nphi=32)[1]))
    ncol = len(boundary_edges(film_cylinder_seed(nphi=32)[1]))
    good = (nb == 2 * 48 and nc == 48 and nf == 64 and ncol == 64)
    ok &= good
    print(f"cmc: rim crease edges bridge {nb} (want 96), cap {nc} "
          f"(want 48), films {nf}/{ncol} (want 64) "
          f"{'OK' if good else 'FAIL'}")

    # wetted-area gradient vs central differences (contact-line rows
    # only, in-plane)
    rng = np.random.default_rng(3)
    loop = rim_loop(Vc, 48)
    d = rng.normal(size=Vc.shape)
    d[:, 2] = 0.0                        # in-plane variation
    h = 1e-6
    fd = (wetted_area(Vc + h * d, loop)
          - wetted_area(Vc - h * d, loop)) / (2 * h)
    an = float(np.sum(wetted_area_grad(Vc, loop) * d))
    good = abs(fd - an) < 1e-8 * max(1.0, abs(fd))
    ok &= good
    print(f"cmc: wetted-area gradient vs FD rel err "
          f"{abs(fd - an) / max(abs(fd), 1e-30):.2e} "
          f"{'OK' if good else 'FAIL'}")
    good = abs(wetted_area(Vc, loop) - math.pi * geo["a"] ** 2) \
        < 5e-3 * math.pi * geo["a"] ** 2
    ok &= good
    print(f"cmc: seed wetted area {wetted_area(Vc, loop):.5f} vs "
          f"pi a^2 {math.pi * geo['a'] ** 2:.5f} "
          f"{'OK' if good else 'FAIL'}")

    # signed mean curvature: +1/r on an outward icosphere.  On the
    # icosahedral combinatorics the vertex estimator's spread is O(h)
    # (it does not converge pointwise at the valence-5 vertices), the
    # mean is O(h^2): assert both behaviours rather than a fantasy
    # pointwise accuracy.
    try:
        from .surfaces.primitives import icosphere
    except ImportError:
        from surfaces.primitives import icosphere    # type: ignore
    stats = []
    for sub in (3, 4):
        SV, SF = icosphere(sub, 'per_level')
        SV = 2.0 * np.asarray(SV, float)
        Hs = signed_mean_curvature(SV, np.asarray(SF, np.int64))
        stats.append((abs(float(np.mean(Hs)) - 0.5), float(np.std(Hs))))
    good = (stats[0][0] < 1e-3 and stats[1][0] < stats[0][0]
            and stats[1][1] < 0.7 * stats[0][1] and stats[1][1] < 6e-3)
    ok &= good
    print(f"cmc: signed H on radius-2 sphere: mean err "
          f"{stats[0][0]:.1e} -> {stats[1][0]:.1e}, std "
          f"{stats[0][1]:.1e} -> {stats[1][1]:.1e} under refinement "
          f"{'OK' if good else 'FAIL'}")

    # perturbed cylinder-volume bridge relaxes back: p ~ 1 = 2H
    V, T, info = relax_bridge(volume_factor=1.0, nring=32, nrow=13,
                              iters=200, squash=0.08)
    p = float(info["pressures"][0])
    Hs = signed_mean_curvature(V, T)[32:-32]
    good = (abs(p - 1.0) < 2e-2
            and abs(p - 2.0 * float(np.mean(Hs))) < 1e-2
            and max(hh["rise"] for hh in info["history"]) <= 1e-12
            and max(hh["drift_post"] for hh in info["history"]) < 1e-10)
    ok &= good
    print(f"cmc: cylinder bridge p={p:.4f} (want ~1), p-2H = "
          f"{p - 2.0 * float(np.mean(Hs)):+.1e}, monotone, drift OK "
          f"{'OK' if good else 'FAIL'}")

    # gravity: the z-moment is exact on the polygonal cap volume of a
    # PRISM slab z in [0, h] over the rim polygon?  Use the closed cap
    # mesh directly: for the analytic spherical cap, int z dV has a
    # closed form; check convergence, and check the analytic gradient
    # against central differences (the FD is the ground truth here).
    geoz = cap_geometry(math.radians(75.0), 1.0)
    Rz, thz = geoz["R"], math.radians(75.0)
    zc = -Rz * math.cos(thz)
    # int z dV over the cap (apex up, base in z=0): integrate slices
    # pi r(z)^2 z dz with r^2 = R^2 - (z - zc)^2, z in [0, height]
    hh_ = geoz["height"]
    Mz_exact = math.pi * (
        (Rz * Rz - zc * zc) * hh_ * hh_ / 2.0
        + 2.0 * zc * hh_ ** 3 / 3.0 - hh_ ** 4 / 4.0)
    errs = []
    for nring in (24, 48):
        Vc, Tc, Lc, rim = build_cap_mesh(75.0, geoz["R"], nring)
        errs.append(abs(z_moment(Vc, Tc, Lc) - Mz_exact) / Mz_exact)
    good = errs[1] < errs[0] / 3.0 and errs[1] < 1.5e-2
    ok &= good
    print(f"cmc: z-moment vs analytic cap {errs[0]:.2e} -> "
          f"{errs[1]:.2e} under refinement {'OK' if good else 'FAIL'}")
    d = rng.normal(size=Vc.shape)
    h = 1e-6
    fd = (z_moment(Vc + h * d, Tc, Lc)
          - z_moment(Vc - h * d, Tc, Lc)) / (2 * h)
    an = float(np.sum(z_moment_grad(Vc, Tc, Lc) * d))
    err = abs(fd - an) / max(abs(fd), 1e-30)
    good = err < 1e-8
    ok &= good
    print(f"cmc: z-moment gradient vs FD rel err {err:.2e} "
          f"{'OK' if good else 'FAIL'}")

    # quadric normal estimator: on a sphere lathe it must beat the raw
    # face-average normals by an order of magnitude
    ids = np.arange(48, 2 * 48)          # an interior ring
    nq = quadric_vertex_normals(Vc, Tc, ids)
    ns = Vc[ids] - np.array([0.0, 0.0, zc])
    ns /= np.linalg.norm(ns, axis=1, keepdims=True)
    devq = float(np.max(np.degrees(np.arccos(np.clip(
        np.einsum('ij,ij->i', nq, ns), -1.0, 1.0)))))
    good = devq < 0.05
    ok &= good
    print(f"cmc: quadric normals on sphere max dev {devq:.3f} deg "
          f"{'OK' if good else 'FAIL'}")

    # free-boundary film smoke: tilted bulged disk in a cylinder must
    # flatten to the cross-section disk and meet the wall orthogonally;
    # the honest discrete reference is the INSCRIBED POLYGON disk (the
    # boundary chords the circle), not pi R^2 -- vs pi the residual is
    # exactly the polygon deficit (2 pi/n)^2/6
    V, T, labels, fixed, freeb, wall = film_disk_seed(1.0, nphi=32)
    info = relax_film(V, T, labels, fixed, wall, freeb, iters=300)
    loop = boundary_loop(T, freeb)
    dev_fit, dev_raw = film_contact_angle_dev(V, T, wall, loop)
    a_err = abs(info["area"] - polygon_area(1.0, 32)) / math.pi
    wres = max(hh["wall_resid"] for hh in info["history"])
    rise = max(hh["rise"] for hh in info["history"] if not hh["groomed"])
    good = (a_err < 1e-3 and float(np.sqrt(np.mean(dev_fit ** 2))) < 0.2
            and wres < 1e-9 and len(info["pressures"]) == 0
            and rise <= 1e-12)
    ok &= good
    print(f"cmc: disk film in cylinder: area vs polygon disk "
          f"{a_err:.1e}, contact dev rms "
          f"{float(np.sqrt(np.mean(dev_fit ** 2))):.3f} deg "
          f"(raw {float(np.sqrt(np.mean(dev_raw ** 2))):.2f}), wall "
          f"resid {wres:.1e} {'OK' if good else 'FAIL'}")

    # sessile drop at 60 degrees from a hemisphere seed: the contact
    # line slides OUT and the achieved angle lands near 60 (the bench
    # measures the tight tolerances; this is the smoke gate)
    V, T, info = relax_drop(theta_deg=60.0, nring=32, iters=400)
    ang, rms = measured_contact_angle(V, T)
    wres = max(hh["wall_resid"] for hh in info["history"])
    erise = max(hh["E_rise"] for hh in info["history"]
                if not hh["groomed"])
    drift = max(hh["drift_post"] for hh in info["history"])
    good = (abs(ang - 60.0) < 1.0 and wres < 1e-9 and erise <= 1e-12
            and drift < 1e-10 and rms < 5e-3)
    ok &= good
    print(f"cmc: sessile drop 60 deg -> achieved {ang:.3f} deg, wall "
          f"resid {wres:.1e}, E monotone (max rise {erise:.1e}), "
          f"drift {drift:.1e} {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("cmc_generator self-test failed")
