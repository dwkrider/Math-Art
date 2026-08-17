# CMC / capillary surface generator: liquid bridges (the Delaunay
# surfaces of revolution -- unduloid and nodoid, with the catenoid and
# the cylinder as limits) and sessile drops with a prescribed contact
# angle, produced by genuine constrained area minimization rather than
# closed-form drawing.
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
# In both cases the equilibrium satisfies the Young-Laplace law
# p = 2 sigma H with CONSTANT mean curvature H; the solver's Lagrange
# multiplier is the pressure, reported per run, and tests/bench
# measures H constancy, p vs 2H, the spherical-cap ground truth and
# the catenoid limit.
#
# Pure-math core (no bpy): mesh builders, volume bookkeeping for open
# surfaces, wetting energy/gradient, and the measurement instruments
# (sphere fit, signed mean curvature) -- all reused by tests/bench.
#
# References:
#   C. E. Delaunay, "Sur la surface de revolution dont la courbure
#       moyenne est constante", Journal de Mathematiques Pures et
#       Appliquees 6 (1841), 309-314 -- the classification of CMC
#       surfaces of revolution (unduloid, nodoid, catenoid, cylinder,
#       sphere).
#   T. Young, "An essay on the cohesion of fluids", Philosophical
#       Transactions of the Royal Society 95 (1805), 65-87 -- the
#       contact-angle condition.
#   P. S. de Laplace, "Traite de mecanique celeste", Supplement au
#       livre X (1806) -- the Young-Laplace pressure law.
#   J. Plateau, "Statique experimentale et theorique des liquides
#       soumis aux seules forces moleculaires" (1873).
#   R. Finn, "Equilibrium Capillary Surfaces", Grundlehren der
#       mathematischen Wissenschaften 284, Springer (1986) -- sessile
#       drops and capillary theory.
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992) -- constrained evolution; wetting as a contact-line
#       integral; level-set constraint walls (cnstrnt.c).

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


def measured_contact_angle(V, T):
    """Contact angle (degrees) of a relaxed drop read from the sphere
    fitted to its free surface: cos(theta) = -z_center / R_fit (the
    floor is z = 0).  Also returns the fit rms."""
    ids = np.unique(np.asarray(T).ravel())
    c, r, rms = sphere_fit(np.asarray(V, float)[ids])
    cosv = max(-1.0, min(1.0, -float(c[2]) / max(r, 1e-300)))
    return math.degrees(math.acos(cosv)), rms


def relax_bridge(volume_factor=1.15, nring=48, nrow=17, h=0.5, R=1.0,
                 iters=600, groom_every=0, squash=0.0):
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
                        fixed=fixed, groom_every=groom_every)
    return V, T, info


def relax_drop(theta_deg=60.0, volume=2.0 * math.pi / 3.0, nring=48,
               iters=1000, groom_every=4, seed_theta_deg=90.0):
    """Build and evolve a sessile drop: contact line sliding on the
    floor plane (two-sided wall on the rim, one-sided on the interior),
    wetting energy -cos(theta) * wetted area, volume constrained.
    The seed is a spherical cap of angle seed_theta_deg at the target
    volume, so the run demonstrably MOVES the contact line to the
    prescribed angle.  Returns (V, T, info)."""
    geo_seed = cap_geometry(math.radians(seed_theta_deg), volume)
    V, T, labels, rim = build_cap_mesh(seed_theta_deg, geo_seed["R"],
                                       nring)
    interior = ~rim
    loop = rim_loop(V, nring)
    ext_e, ext_g = drop_energy_terms(math.radians(theta_deg), loop)
    plane = _swalls.PlaneWall([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    floor = _swalls.PlaneWall([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                              one_sided=True)
    info = _svol.evolve(V, T, labels, targets=[volume], iters=iters,
                        groom_every=groom_every,
                        walls=[(plane, rim), (floor, interior)],
                        ext_energy=ext_e, ext_grad=ext_g)
    return V, T, info


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

if _IN_BLENDER:

    class MESH_OT_cmc_capillary_add(bpy.types.Operator):
        """Constant-mean-curvature capillary surface by constrained
        evolution: a liquid bridge between two rings (Delaunay's
        unduloid/nodoid family, catenoid at the right volume) or a
        sessile drop with a prescribed contact angle on a floor"""
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
                    "contact line sliding to Young's contact angle")],
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
        resolution: IntProperty(
            name="Resolution", default=48, min=16, max=128,
            description="Vertices around the rim / contact line")
        iterations: IntProperty(
            name="Evolve Iterations", default=800, min=0, max=5000,
            description="Constrained area-descent iterations "
                        "(0 shows the raw seed)")
        smooth: BoolProperty(name="Smooth Shading", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            if self.mode == 'BRIDGE':
                nrow = max(5, 2 * int(round(
                    self.aspect * self.resolution / (2.0 * math.pi)))
                    * 2 + 1)
                V, T, info = relax_bridge(
                    volume_factor=self.volume_factor,
                    nring=self.resolution, nrow=nrow,
                    h=self.aspect, R=1.0,
                    iters=self.iterations)
                name = "Liquid Bridge"
                Hs = signed_mean_curvature(V, T)
                # interior rows only (rims are pinned)
                nring = self.resolution
                Hmean = float(np.mean(Hs[nring:-nring])) \
                    if len(V) > 2 * nring else 0.0
                extra = f"H={Hmean:.4f}"
            else:
                V, T, info = relax_drop(
                    theta_deg=self.contact_angle,
                    nring=self.resolution,
                    iters=self.iterations)
                name = "Sessile Drop"
                ang, _rms = measured_contact_angle(V, T)
                extra = f"contact angle {ang:.2f} deg " \
                        f"(asked {self.contact_angle:.1f})"

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
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            p = float(info["pressures"][0])
            self.report({'INFO'},
                        f"{name}: {info['iters_run']} iterations, "
                        f"area {info['area']:.4f}, pressure {p:.4f}, "
                        f"{extra}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'BRIDGE':
                lay.prop(self, 'volume_factor')
                lay.prop(self, 'aspect')
            else:
                lay.prop(self, 'contact_angle')
            for k in ('resolution', 'iterations', 'smooth', 'scale'):
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
