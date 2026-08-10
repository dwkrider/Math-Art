
# Lorenz Manifold Generator for Blender
#
# The two-dimensional STABLE MANIFOLD of the origin of the Lorenz
# system -- the surface of all points that flow into the equilibrium at
# 0 rather than onto the butterfly attractor. It is the object Osinga
# and Krauskopf crocheted, and the one the 2026 Alternative Fields
# Medal trophies were printed from.
#
#   x' = sigma (y - x),   y' = x (rho - z) - y,   z' = x y - beta z
#
# At the origin the Jacobian has eigenvalues +11.83, -22.83 and -8/3;
# the two negative ones span the stable eigenspace, and the manifold is
# tangent to it there.  Growing it outward means integrating BACKWARD
# in time, and that is where the difficulty lies: the two stable rates
# differ by a factor of 8.6, so a front of equal-TIME images stretches
# by 8.6^t between the fast and slow directions and the mesh degenerates
# within a few units.
#
# The cure used here is to integrate at UNIT SPEED,
#
#     dx/dtau = -f(x) / |f(x)| ,
#
# so every trajectory advances the same ARCLENGTH rather than the same
# time, and to resample the ring to evenly spaced points every so often.
# This is the method of the AMS Notices article below, which is how the
# printed trophies were made; the residual tangential bunching is what
# the resampling removes.  The alternative -- Krauskopf and Osinga's
# geodesic level sets, where each new ring point solves a two-unknown
# boundary value problem -- produces true geodesic circles and reaches
# much larger geodesic radii, at a cost this module does not need.
#
# The local seed is the LINEAR stable eigenspace at radius delta rather
# than a high-order parameterization of the manifold.  The error is
# O(delta^2), which at delta = 0.02 is far below the mesh resolution
# once the result is fitted into a 2 m cube; the parameterization
# method is the upgrade if that ever stops being true.
#
# Geometry only; materials and rendering are left to Blender.
#
# References:
# - The system: E. N. Lorenz, "Deterministic nonperiodic flow",
#   Journal of the Atmospheric Sciences 20(2), 1963, pp. 130-141.
# - Geodesic level-set algorithm: B. Krauskopf and H. M. Osinga,
#   "Computing geodesic level sets on global (un)stable manifolds of
#   vector fields", SIAM Journal on Applied Dynamical Systems 2(4),
#   2003, pp. 546-569.
# - The manifold as an object to make: H. M. Osinga and B. Krauskopf,
#   "Crocheting the Lorenz manifold", The Mathematical Intelligencer
#   26(4), 2004, pp. 25-37.
# - The arclength-reparametrized growth used here, and the
#   parameterization method for the local piece: P. R. Bishop,
#   S. Chenoweth, E. Fleurantin, A. Ogueda-Oliva, E. Sander and
#   J. Seay, "3D printing of invariant manifolds in dynamical systems",
#   Notices of the American Mathematical Society, 2026;
#   preprint arXiv:2504.15884.

bl_info = {
    "name": "Lorenz Manifold Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Lorenz Manifold",
    "description": "The two-dimensional stable manifold of the origin "
                   "of the Lorenz system, grown at unit speed",
    "category": "Add Mesh",
}

import math

import numpy as np


# ==========================================================================
# The vector field and its linearisation at the origin
# ==========================================================================

def lorenz_field(X, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """The Lorenz vector field on an (n, 3) array of states."""
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z], axis=-1)


def lorenz_jacobian(sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """J(0): the linearisation at the origin, where the manifold is
    tangent to the stable eigenspace."""
    return np.array([[-sigma, sigma, 0.0],
                     [rho, -1.0, 0.0],
                     [0.0, 0.0, -beta]])


def stable_eigenbasis(sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """An orthonormal basis of the two-dimensional stable eigenspace of
    the origin, plus the eigenvalues.  Raises if the equilibrium is not
    of saddle type with a two-dimensional stable subspace."""
    w, V = np.linalg.eig(lorenz_jacobian(sigma, rho, beta))
    if np.max(np.abs(w.imag)) > 1e-12:
        raise ValueError("the origin has complex eigenvalues here; this "
                         "generator wants a real saddle")
    w = w.real
    V = V.real
    neg = np.nonzero(w < 0.0)[0]
    if len(neg) != 2:
        raise ValueError(f"the origin needs exactly two stable "
                         f"directions, found {len(neg)} "
                         f"(eigenvalues {np.round(w, 4)})")
    # Gram-Schmidt: the eigenvectors span the plane but are not
    # orthogonal, and an even seed circle wants an orthonormal frame
    a = V[:, neg[0]]
    a = a / np.linalg.norm(a)
    b = V[:, neg[1]]
    b = b - np.dot(b, a) * a
    b = b / np.linalg.norm(b)
    return a, b, w[neg], w[w > 0.0]


def nontrivial_equilibria(sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """C+ and C-, the two equilibria the manifold spirals around but
    never contains.  The field vanishes there, so unit-speed
    integration has to be told about them."""
    if rho <= 1.0:
        return np.zeros((0, 3))
    r = math.sqrt(beta * (rho - 1.0))
    return np.array([[r, r, rho - 1.0], [-r, -r, rho - 1.0]])


# ==========================================================================
# Growing the manifold
# ==========================================================================

def resample_closed(P, n, cubic=True):
    """Resample a closed polyline to n evenly spaced points by
    cumulative chord length.  The ring is closed, so the wrap segment
    counts.

    Interpolation is Catmull-Rom by default, not linear.  This is the
    accuracy bottleneck of the whole method: every resampling step
    moves each point off the true ring by the chord-to-arc deviation,
    and those errors are what the forward flow later amplifies.  Going
    from linear to cubic drops that deviation from O(h^2) to O(h^4) for
    free -- measurably, it roughly halves how far the finished surface
    sits off the manifold, at no cost in ring count."""
    P = np.asarray(P, dtype=float)
    m = len(P)
    ring = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0.0:
        raise ValueError("the ring collapsed to a point")
    t = np.linspace(0.0, total, int(n), endpoint=False)
    idx = np.clip(np.searchsorted(cum, t, side='right') - 1, 0, m - 1)
    u = ((t - cum[idx]) / np.maximum(seg[idx], 1e-300))[:, None]
    if not cubic or m < 4:
        return ring[idx] + u * (ring[idx + 1] - ring[idx])
    p0 = P[(idx - 1) % m]
    p1 = P[idx % m]
    p2 = P[(idx + 1) % m]
    p3 = P[(idx + 2) % m]
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (2.0 * p1
                  + (-p0 + p2) * u
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3)


def grow_manifold(sigma=10.0, rho=28.0, beta=8.0 / 3.0, arclength=100.0,
                  seed_radius=0.02, seed_points=96, ring_spacing=1.0,
                  target_edge=0.25, max_ring_points=3000, step=0.02,
                  cubic=True):
    """Grow the stable manifold of the origin outward and return the
    list of rings, in the system's own coordinates.

    Every point advances the same arclength per step, which is what
    keeps the fast and slow stable directions in step with each other.
    Between checkpoints the ring is resampled to evenly spaced points
    and allowed to gain points as it lengthens."""
    u1, u2, w_s, w_u = stable_eigenbasis(sigma, rho, beta)
    n0 = max(8, int(seed_points))
    ang = 2.0 * math.pi * np.arange(n0) / n0
    ring = (float(seed_radius)
            * (np.cos(ang)[:, None] * u1 + np.sin(ang)[:, None] * u2))

    centres = nontrivial_equilibria(sigma, rho, beta)

    def unit_backward(X):
        # the STABLE manifold grows backward in time; with the unit
        # speed normalisation that is -f/|f|, and the sign is the one
        # thing here that silently produces nonsense if flipped
        F = lorenz_field(X, sigma, rho, beta)
        nrm = np.linalg.norm(F, axis=-1, keepdims=True)
        return -F / np.maximum(nrm, 1e-9)

    def rk4(X, h):
        k1 = unit_backward(X)
        k2 = unit_backward(X + 0.5 * h * k1)
        k3 = unit_backward(X + 0.5 * h * k2)
        k4 = unit_backward(X + h * k3)
        return X + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    rings = [ring.copy()]
    h = float(step)
    total = float(arclength)
    gap = max(float(ring_spacing), h)
    done = 0.0
    near_equilibrium = 0
    while done < total - 1e-12:
        target = min(done + gap, total)
        while done < target - 1e-12:
            ring = rk4(ring, min(h, target - done))
            done += min(h, target - done)
        if len(centres):
            d = np.min(np.linalg.norm(ring[:, None, :]
                                      - centres[None, :, :], axis=-1))
            if d < 0.05:
                near_equilibrium += 1
        # re-parametrize: even spacing, and more points as it lengthens
        seg = np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]),
                                     axis=0), axis=1)
        length = float(seg.sum())
        want = int(np.clip(round(length / max(target_edge, 1e-6)),
                           n0, int(max_ring_points)))
        ring = resample_closed(ring, want, cubic=cubic)
        rings.append(ring.copy())
    return rings, {'eigenvalues_stable': w_s, 'eigenvalue_unstable': w_u,
                   'near_equilibrium': near_equilibrium,
                   'arclength': total}


def mesh_rings(rings):
    """Stitch consecutive rings into a triangle mesh, capping the
    innermost ring with a fan to the origin.

    Neighbouring rings generally have different point counts, so the
    band between them is walked with two indices advancing on
    NORMALISED arclength -- pairing by index would shear the mesh as
    soon as one ring gained points."""
    verts = [np.zeros(3)]
    base = [1]
    for r in rings:
        base.append(base[-1] + len(r))
    offs = [1]
    for r in rings[:-1]:
        offs.append(offs[-1] + len(r))
    for r in rings:
        verts.extend(list(r))
    faces = []
    # cap: fan from the equilibrium to the innermost ring
    n_in = len(rings[0])
    for i in range(n_in):
        faces.append((0, offs[0] + i, offs[0] + (i + 1) % n_in))
    for k in range(len(rings) - 1):
        a0, a1 = offs[k], offs[k + 1]
        na, nb = len(rings[k]), len(rings[k + 1])
        i = j = 0
        while i < na or j < nb:
            ta = (i + 1) / na if i < na else 2.0
            tb = (j + 1) / nb if j < nb else 2.0
            if ta <= tb:
                faces.append((a0 + i % na, a1 + j % nb,
                              a0 + (i + 1) % na))
                i += 1
            else:
                faces.append((a0 + i % na, a1 + j % nb,
                              a1 + (j + 1) % nb))
                j += 1
    return np.asarray(verts, dtype=float), faces


def center_fit(verts, scale=1.0):
    """Centre on the bounding box and fit the largest extent to a 2 m
    cube (the project-wide convention), then apply `scale`."""
    verts = np.asarray(verts, dtype=float)
    if not len(verts):
        return verts
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ext = float((hi - lo).max())
    return (verts - 0.5 * (lo + hi)) * (2.0 / ext
                                        if ext > 1e-9 else 1.0) * scale


def build_lorenz_manifold(sigma=10.0, rho=28.0, beta=8.0 / 3.0,
                          arclength=100.0, seed_radius=0.02,
                          seed_points=96, ring_spacing=1.0,
                          target_edge=0.25, max_ring_points=3000,
                          step=0.02, scale=1.0):
    """Mesh the stable manifold.  Returns (verts, faces, info)."""
    rings, info = grow_manifold(
        sigma=sigma, rho=rho, beta=beta, arclength=arclength,
        seed_radius=seed_radius, seed_points=seed_points,
        ring_spacing=ring_spacing, target_edge=target_edge,
        max_ring_points=max_ring_points, step=step)
    verts, faces = mesh_rings(rings)
    info.update({'rings': len(rings), 'verts': len(verts),
                 'faces': len(faces),
                 'outer_points': len(rings[-1])})
    return center_fit(verts, scale), faces, info


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

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

    class MESH_OT_lorenz_manifold_add(bpy.types.Operator):
        """Add the two-dimensional stable manifold of the origin of the
        Lorenz system, grown outward at unit speed"""
        bl_idname = "mesh.lorenz_manifold_add"
        bl_label = "Lorenz Manifold"
        bl_options = {'REGISTER', 'UNDO'}

        sigma: FloatProperty(
            name="Sigma", default=10.0, min=0.1, max=50.0,
            description="Prandtl number; 10 is Lorenz's own value")
        rho: FloatProperty(
            name="Rho", default=28.0, min=1.1, max=100.0,
            description="Rayleigh number; 28 is Lorenz's own value")
        beta: FloatProperty(
            name="Beta", default=8.0 / 3.0, min=0.1, max=10.0,
            description="Geometric factor; 8/3 is Lorenz's own value")
        arclength: FloatProperty(
            name="Arclength", default=100.0, min=5.0, max=250.0,
            description="How far to grow the manifold, measured along "
                        "trajectories rather than in time")
        seed_radius: FloatProperty(
            name="Seed Radius", default=0.02, min=1e-4, max=0.5,
            description="Radius of the starting circle in the stable "
                        "eigenspace; the linear seed costs an error of "
                        "order its square")
        seed_points: IntProperty(
            name="Seed Points", default=96, min=16, max=512,
            description="Points on the starting circle")
        ring_spacing: FloatProperty(
            name="Ring Spacing", default=1.0, min=0.1, max=5.0,
            description="Arclength between recorded rings")
        target_edge: FloatProperty(
            name="Target Edge", default=0.25, min=0.02, max=2.0,
            description="Wanted spacing along a ring; smaller keeps "
                        "the sharp folds from being cut across")
        max_ring_points: IntProperty(
            name="Max Ring Points", default=3000, min=200, max=20000,
            description="Cap on the points in one ring")
        step: FloatProperty(
            name="Step", default=0.02, min=0.002, max=0.2,
            description="Arclength step of the integrator")
        thickness: FloatProperty(
            name="Thickness", default=0.03, min=0.0, max=0.5,
            description="If > 0, add a Solidify modifier -- the "
                        "manifold is a surface, so it needs a shell to "
                        "print")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            try:
                verts, faces, info = build_lorenz_manifold(
                    sigma=self.sigma, rho=self.rho, beta=self.beta,
                    arclength=self.arclength,
                    seed_radius=self.seed_radius,
                    seed_points=self.seed_points,
                    ring_spacing=self.ring_spacing,
                    target_edge=self.target_edge,
                    max_ring_points=self.max_ring_points,
                    step=self.step, scale=self.scale)
            except (ValueError, np.linalg.LinAlgError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

            obj = _new_object(context, "Lorenz Manifold", verts, faces,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            me = obj.data
            if info.get('near_equilibrium'):
                self.report(
                    {'WARNING'},
                    f"the ring passed within 0.05 of C+ or C- on "
                    f"{info['near_equilibrium']} checkpoints, where the "
                    f"field nearly vanishes; the growth there is less "
                    f"reliable")
            if info.get('outer_points', 0) >= self.max_ring_points:
                self.report(
                    {'WARNING'},
                    f"the outermost ring hit the {self.max_ring_points}"
                    f"-point cap, so it is under-resolved -- raise Max "
                    f"Ring Points or lower Arclength")
            ws = np.round(info['eigenvalues_stable'], 3)
            self.report({'INFO'},
                        f"Lorenz manifold: {info['rings']} rings, "
                        f"arclength {info['arclength']:.0f}, "
                        f"{len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces; stable eigenvalues "
                        f"{ws}")
            return {'FINISHED'}

        def draw(self, lay_context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('sigma', 'rho', 'beta'):
                lay.prop(self, k)
            lay.separator()
            for k in ('arclength', 'ring_spacing', 'target_edge',
                      'seed_radius', 'seed_points', 'max_ring_points',
                      'step'):
                lay.prop(self, k)
            lay.separator()
            for k in ('thickness', 'scale', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.lorenz_manifold_add",
                             icon='SURFACE_NSURFACE')

    _classes = (MESH_OT_lorenz_manifold_add,)

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

def _edge_counts(faces):
    cnt = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    return cnt


def _selftest():
    # ---- 1. the linearisation at the origin -------------------------
    u1, u2, w_s, w_u = stable_eigenbasis()
    if abs(float(np.max(w_u)) - 11.8277) > 1e-3:
        raise AssertionError(f"unstable eigenvalue {w_u}, expected "
                             f"+11.8277")
    want = sorted([-22.8277, -8.0 / 3.0])
    got = sorted(float(v) for v in w_s)
    if any(abs(a - b) > 1e-3 for a, b in zip(got, want)):
        raise AssertionError(f"stable eigenvalues {got}, expected "
                             f"{want}")
    if (abs(float(np.dot(u1, u2))) > 1e-12
            or abs(float(np.linalg.norm(u1)) - 1.0) > 1e-12
            or abs(float(np.linalg.norm(u2)) - 1.0) > 1e-12):
        raise AssertionError("the stable eigenbasis is not orthonormal")
    ratio = abs(min(got)) / abs(max(got))
    print(f"origin: stable eigenvalues {np.round(got, 4)}, unstable "
          f"{float(np.max(w_u)):.4f}; the stable rates differ by "
          f"{ratio:.1f}x, which is why equal-time growth fails")

    # the seed circle must lie in the stable eigenspace: J maps that
    # plane to itself
    J = lorenz_jacobian()
    nrm = np.cross(u1, u2)
    for v in (u1, u2):
        if abs(float(np.dot(J @ v, nrm))) > 1e-9:
            raise AssertionError("the eigenbasis does not span an "
                                 "invariant plane of J")

    # ---- 2. the growth ----------------------------------------------
    rings, info = grow_manifold(arclength=30.0, ring_spacing=1.0,
                                target_edge=0.25)
    if len(rings) < 5:
        raise AssertionError(f"only {len(rings)} rings")
    lengths = []
    for r in rings:
        seg = np.linalg.norm(np.diff(np.vstack([r, r[:1]]), axis=0),
                             axis=1)
        lengths.append(float(seg.sum()))
    if any(b <= a for a, b in zip(lengths[:-1], lengths[1:])):
        raise AssertionError("the rings do not grow monotonically: "
                             f"{np.round(lengths, 3)}")
    counts = [len(r) for r in rings]
    if any(b < a for a, b in zip(counts[:-1], counts[1:])):
        raise AssertionError(f"ring point counts fell: {counts}")
    print(f"growth: {len(rings)} rings, circumference "
          f"{lengths[0]:.3f} -> {lengths[-1]:.1f}, points "
          f"{counts[0]} -> {counts[-1]}")

    # ---- 3. it really is the STABLE manifold ------------------------
    # Points of a stable manifold flow INTO the equilibrium.  They
    # cannot stay there numerically: the origin is a saddle, so any
    # deviation is amplified by e^(11.83 t), and a point sitting a
    # little off the surface dives toward 0 and then shoots away.  So
    # the measure is the CLOSEST APPROACH, as a fraction of the
    # starting radius, and it is only meaningful next to a control --
    # generic nearby points must do markedly worse, or the test would
    # pass on any surface in the neighbourhood.
    def _closest_approach(S, tmax=3.0, dt=0.002):
        X = np.asarray(S, dtype=float).copy()
        best = np.linalg.norm(X, axis=1).copy()
        for _ in range(int(tmax / dt)):
            k1 = lorenz_field(X)
            k2 = lorenz_field(X + 0.5 * dt * k1)
            k3 = lorenz_field(X + 0.5 * dt * k2)
            k4 = lorenz_field(X + dt * k3)
            X = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            best = np.minimum(best, np.linalg.norm(X, axis=1))
        return best / np.maximum(np.linalg.norm(S, axis=1), 1e-12)

    outer = rings[-1]
    sample = outer[np.linspace(0, len(outer) - 1, 16).astype(int)]
    on = _closest_approach(sample)
    rng = np.random.default_rng(0)
    off = _closest_approach(sample + 0.5 * rng.normal(size=sample.shape))
    if float(np.median(on)) > 0.10 or float(np.max(on)) > 0.25:
        raise AssertionError(
            f"points of the surface should dive close to the origin "
            f"under the forward flow, but reached only "
            f"{float(np.median(on)):.3f} of their radius (worst "
            f"{float(np.max(on)):.3f}) -- this is not the stable "
            f"manifold")
    if float(np.median(off)) < 4.0 * float(np.median(on)):
        raise AssertionError(
            f"points nudged 0.5 off the surface reached "
            f"{float(np.median(off)):.3f} against the surface's "
            f"{float(np.median(on)):.3f}; the test does not "
            f"discriminate")
    print(f"invariance: the outer ring dives to "
          f"{float(np.median(on)):.3f} of its radius (worst "
          f"{float(np.max(on)):.3f}); points nudged 0.5 off reach only "
          f"{float(np.median(off)):.3f}")

    # cubic resampling is what buys that accuracy -- linear
    # interpolation leaves the surface measurably further off
    lin, _ = grow_manifold(arclength=30.0, cubic=False)
    lo = lin[-1]
    lin_on = _closest_approach(
        lo[np.linspace(0, len(lo) - 1, 16).astype(int)])
    if float(np.median(lin_on)) <= float(np.median(on)):
        raise AssertionError(
            f"cubic resampling should beat linear, but linear reached "
            f"{float(np.median(lin_on)):.3f} against "
            f"{float(np.median(on)):.3f}")
    print(f"resampling: Catmull-Rom {float(np.median(on)):.3f} vs "
          f"linear {float(np.median(lin_on)):.3f} -- the interpolation "
          f"is the accuracy bottleneck, not the integrator")

    # ---- 4. the mesh ------------------------------------------------
    V, F, info = build_lorenz_manifold(arclength=30.0)
    cnt = _edge_counts(F)
    boundary = [e for e, c in cnt.items() if c == 1]
    over = [e for e, c in cnt.items() if c > 2]
    if over:
        raise AssertionError(f"{len(over)} edges have more than two "
                             f"faces; the band stitching is wrong")
    chi = len(V) - len(cnt) + len(F)
    if chi != 1:
        raise AssertionError(f"the manifold is a disk, so chi should "
                             f"be 1, got {chi}")
    # the single boundary is the outermost ring
    if len(boundary) != info['outer_points']:
        raise AssertionError(
            f"{len(boundary)} boundary edges against an outer ring of "
            f"{info['outer_points']} points -- the free rim should be "
            f"exactly that ring")
    if not np.all(np.isfinite(V)):
        raise AssertionError("non-finite vertices")
    ext = float((V.max(axis=0) - V.min(axis=0)).max())
    if abs(ext - 2.0) > 1e-6:
        raise AssertionError(f"{ext:.4f} across, expected a 2 m fit")
    print(f"mesh: chi {chi}, one rim of {len(boundary)} edges, "
          f"{len(V)} verts, {len(F)} faces")

    # ---- 5. the sign of the integration -----------------------------
    # Growing a stable manifold means integrating BACKWARD.  Check the
    # seed ring moves away from the origin, not into it.
    r0, _ = grow_manifold(arclength=2.0, ring_spacing=2.0)
    if (float(np.mean(np.linalg.norm(r0[-1], axis=1)))
            <= float(np.mean(np.linalg.norm(r0[0], axis=1)))):
        raise AssertionError("the ring did not move outward; the "
                             "integration sign is flipped")
    print("RESULT: OK")
