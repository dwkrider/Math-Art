
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
# Systems, their equilibria and the eigenspace at one of them
# ==========================================================================
# A system is a right-hand side plus a list of equilibria in closed
# form.  The Jacobian is taken numerically by central differences --
# accurate to about 1e-10, which is far tighter than anything here
# needs, and it means adding a system is a matter of writing down f and
# solving f = 0 rather than hand-differentiating a matrix.  The
# self-test checks the numeric Jacobian against Lorenz's analytic one.

def lorenz_field(X, p):
    """Lorenz, "Deterministic nonperiodic flow" (1963).  p = (sigma,
    rho, beta)."""
    sigma, rho, beta = p
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z], axis=-1)


def lorenz_equilibria(p):
    """The origin, always; and the pair C+ / C- once rho > 1, which is
    where the butterfly's two wings are centred."""
    sigma, rho, beta = p
    out = [("Origin", np.zeros(3))]
    if rho > 1.0:
        r = math.sqrt(beta * (rho - 1.0))
        out.append(("C+", np.array([r, r, rho - 1.0])))
        out.append(("C-", np.array([-r, -r, rho - 1.0])))
    return out


def lorenz_jacobian_exact(X, p):
    """The analytic Jacobian, kept only so the self-test can check the
    numeric one against it."""
    sigma, rho, beta = p
    x, y, z = X
    return np.array([[-sigma, sigma, 0.0],
                     [rho - z, -1.0, -x],
                     [y, x, -beta]])


def rossler_field(X, p):
    """Roessler, "An equation for continuous chaos" (1976).
    p = (a, b, c)."""
    a, b, c = p
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([-y - z, x + a * y, b + z * (x - c)], axis=-1)


def rossler_equilibria(p):
    """x' = 0 gives z = -y, y' = 0 gives x = -a y, and z' = 0 then
    reduces to a y^2 + c y + b = 0 -- so there are two equilibria when
    c^2 >= 4ab, an inner one near the origin and a distant outer one."""
    a, b, c = p
    disc = c * c - 4.0 * a * b
    if disc < 0.0 or abs(a) < 1e-12:
        return []
    s = math.sqrt(disc)
    out = []
    for label, y in (("Inner", (-c + s) / (2.0 * a)),
                     ("Outer", (-c - s) / (2.0 * a))):
        out.append((label, np.array([-a * y, y, -y])))
    return out


SYSTEMS = {
    'LORENZ': ("Lorenz", lorenz_field, lorenz_equilibria,
               ("Sigma", "Rho", "Beta"), (10.0, 28.0, 8.0 / 3.0)),
    'ROSSLER': ("Roessler", rossler_field, rossler_equilibria,
                ("a", "b", "c"), (0.2, 0.2, 5.7)),
}


def numeric_jacobian(field, X, p, h=1e-6):
    """Central-difference Jacobian at a single point."""
    X = np.asarray(X, dtype=float)
    J = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        J[:, j] = (field(X + e, p) - field(X - e, p)) / (2.0 * h)
    return J


def invariant_eigenbasis(field, point, p, kind='STABLE'):
    """An orthonormal basis of the two-dimensional invariant eigenspace
    at `point`, plus the eigenvalues that span it.

    The two eigenvalues may be a real pair or a COMPLEX CONJUGATE pair.
    The complex case is the interesting one and the earlier version of
    this module refused it outright: at a spiral saddle -- Lorenz's C+
    and C- are exactly that, with eigenvalues -13.85 and
    0.094 +- 10.195i -- the invariant plane is spanned by the real and
    imaginary parts of one complex eigenvector, and a seed circle in it
    works exactly as in the real case."""
    J = numeric_jacobian(field, point, p)
    w, V = np.linalg.eig(J)
    want = (w.real < 0.0) if kind == 'STABLE' else (w.real > 0.0)
    idx = np.nonzero(want)[0]
    side = "stable" if kind == 'STABLE' else "unstable"
    if len(idx) == 0:
        raise ValueError(f"this equilibrium has no {side} directions")
    if len(idx) == 1:
        raise ValueError(
            f"the {side} manifold of this equilibrium is "
            f"one-dimensional -- a curve, not a surface. Try the "
            f"other side, or another equilibrium")
    if len(idx) > 2:
        raise ValueError(
            f"this equilibrium has {len(idx)} {side} directions, so "
            f"its {side} manifold is the whole space rather than a "
            f"surface")
    sel = w[idx]
    if np.max(np.abs(sel.imag)) > 1e-9:
        # a complex pair: Re(v) and Im(v) span the invariant plane
        v = V[:, idx[0]]
        a, b = v.real.copy(), v.imag.copy()
    else:
        a = V[:, idx[0]].real.copy()
        b = V[:, idx[1]].real.copy()
    na = np.linalg.norm(a)
    if na < 1e-12:
        a, b = b, a
        na = np.linalg.norm(a)
    a = a / na
    b = b - np.dot(b, a) * a
    nb = np.linalg.norm(b)
    if nb < 1e-12:
        raise ValueError("the eigenvectors do not span a plane")
    b = b / nb
    return a, b, sel


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


def grow_manifold(system='LORENZ', params=None, equilibrium=0,
                  kind='STABLE', arclength=100.0, seed_radius=0.02,
                  seed_points=96, ring_spacing=1.0, target_edge=0.25,
                  max_ring_points=3000, step=0.02, cubic=True):
    """Grow a two-dimensional invariant manifold outward from an
    equilibrium and return its rings, in the system's own coordinates.

    Every point advances the same arclength per step, which is what
    keeps fast and slow directions in step with each other.  A STABLE
    manifold is grown by integrating BACKWARD, an unstable one forward;
    that sign is the whole difference between the two, and getting it
    wrong yields the other manifold rather than an error."""
    label, field, equilibria, _names, defaults = SYSTEMS[system]
    p = tuple(defaults if params is None else params)
    eqs = equilibria(p)
    if not eqs:
        raise ValueError(f"{label} has no equilibria at these "
                         f"parameters")
    ei = int(np.clip(int(equilibrium), 0, len(eqs) - 1))
    eq_name, eq = eqs[ei]
    u1, u2, sel = invariant_eigenbasis(field, eq, p, kind)

    n0 = max(8, int(seed_points))
    ang = 2.0 * math.pi * np.arange(n0) / n0
    ring = eq + (float(seed_radius)
                 * (np.cos(ang)[:, None] * u1
                    + np.sin(ang)[:, None] * u2))

    others = np.array([q for nm, q in eqs
                       if not np.allclose(q, eq)], dtype=float)
    sign = -1.0 if kind == 'STABLE' else 1.0

    def unit_flow(X):
        F = field(X, p)
        nrm = np.linalg.norm(F, axis=-1, keepdims=True)
        return sign * F / np.maximum(nrm, 1e-9)

    def rk4(X, h):
        k1 = unit_flow(X)
        k2 = unit_flow(X + 0.5 * h * k1)
        k3 = unit_flow(X + 0.5 * h * k2)
        k4 = unit_flow(X + h * k3)
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
        if len(others):
            d = np.min(np.linalg.norm(ring[:, None, :]
                                      - others[None, :, :], axis=-1))
            if d < 0.05:
                near_equilibrium += 1
        seg = np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]),
                                     axis=0), axis=1)
        length = float(seg.sum())
        want = int(np.clip(round(length / max(target_edge, 1e-6)),
                           n0, int(max_ring_points)))
        ring = resample_closed(ring, want, cubic=cubic)
        rings.append(ring.copy())
    return rings, {'eigenvalues': sel, 'kind': kind,
                   'equilibrium': eq_name, 'equilibrium_point': eq,
                   'system': label, 'params': p,
                   'spiral': bool(np.max(np.abs(sel.imag)) > 1e-9),
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


def build_invariant_manifold(system='LORENZ', params=None,
                             equilibrium=0, kind='STABLE',
                             arclength=100.0, seed_radius=0.02,
                             seed_points=96, ring_spacing=1.0,
                             target_edge=0.25, max_ring_points=3000,
                             step=0.02, scale=1.0):
    """Mesh a two-dimensional invariant manifold.  Returns
    (verts, faces, info)."""
    rings, info = grow_manifold(
        system=system, params=params, equilibrium=equilibrium,
        kind=kind, arclength=arclength, seed_radius=seed_radius,
        seed_points=seed_points, ring_spacing=ring_spacing,
        target_edge=target_edge, max_ring_points=max_ring_points,
        step=step)
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
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    # dynamic enum items are handed to C without their Python strings
    # being kept alive, so anything returned here has to be cached
    _ENUM_CACHE = {}

    def _equilibrium_items(self, context):
        key = getattr(self, 'system', 'LORENZ')
        equilibria = SYSTEMS[key][2]
        try:
            eqs = equilibria((self.p1, self.p2, self.p3))
        except Exception:
            eqs = []
        if not eqs:
            eqs = [("(none)", np.zeros(3))]
        items = [(str(i), nm,
                  f"{nm} at ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})")
                 for i, (nm, q) in enumerate(eqs)]
        _ENUM_CACHE[key] = items
        return items

    def _on_system(self, context):
        """Loading a system loads its own default parameters --
        Lorenz's 10 / 28 / 8/3 mean nothing to Roessler."""
        self.p1, self.p2, self.p3 = SYSTEMS[self.system][4]

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

    class MESH_OT_invariant_manifold_add(bpy.types.Operator):
        """Add a two-dimensional invariant manifold of an equilibrium
        of a chaotic system, grown outward at unit speed"""
        bl_idname = "mesh.invariant_manifold_add"
        bl_label = "Invariant Manifold"
        bl_options = {'REGISTER', 'UNDO'}

        system: EnumProperty(
            name="System",
            items=[(k, v[0], v[0]) for k, v in SYSTEMS.items()],
            default='LORENZ', update=_on_system)
        p1: FloatProperty(name="Parameter 1", default=10.0,
                          min=-100.0, max=100.0)
        p2: FloatProperty(name="Parameter 2", default=28.0,
                          min=-100.0, max=100.0)
        p3: FloatProperty(name="Parameter 3", default=8.0 / 3.0,
                          min=-100.0, max=100.0)
        equilibrium: EnumProperty(name="Equilibrium",
                                  items=_equilibrium_items)
        kind: EnumProperty(
            name="Manifold",
            items=[('STABLE', "Stable", "Points that flow INTO the "
                                        "equilibrium; grown by "
                                        "integrating backward"),
                   ('UNSTABLE', "Unstable", "Points that flow OUT of "
                                            "it; grown forward")],
            default='STABLE')
        arclength: FloatProperty(
            name="Arclength", default=100.0, min=5.0, max=250.0,
            description="How far to grow, measured along trajectories "
                        "rather than in time")
        seed_radius: FloatProperty(
            name="Seed Radius", default=0.02, min=1e-4, max=0.5,
            description="Radius of the starting circle in the "
                        "invariant eigenplane; the linear seed costs "
                        "an error of order its square")
        seed_points: IntProperty(
            name="Seed Points", default=96, min=16, max=512)
        ring_spacing: FloatProperty(
            name="Ring Spacing", default=1.0, min=0.1, max=5.0,
            description="Arclength between recorded rings")
        target_edge: FloatProperty(
            name="Target Edge", default=0.25, min=0.02, max=2.0,
            description="Wanted spacing along a ring; smaller keeps "
                        "the sharp folds from being cut across")
        max_ring_points: IntProperty(
            name="Max Ring Points", default=3000, min=200, max=20000)
        step: FloatProperty(
            name="Step", default=0.02, min=0.002, max=0.2,
            description="Arclength step of the integrator")
        thickness: FloatProperty(
            name="Thickness", default=0.03, min=0.0, max=0.5,
            description="If > 0, add a Solidify modifier -- these are "
                        "surfaces, so they need a shell to print")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            try:
                verts, faces, info = build_invariant_manifold(
                    system=self.system,
                    params=(self.p1, self.p2, self.p3),
                    equilibrium=int(self.equilibrium),
                    kind=self.kind, arclength=self.arclength,
                    seed_radius=self.seed_radius,
                    seed_points=self.seed_points,
                    ring_spacing=self.ring_spacing,
                    target_edge=self.target_edge,
                    max_ring_points=self.max_ring_points,
                    step=self.step, scale=self.scale)
            except (ValueError, np.linalg.LinAlgError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

            side = "Stable" if self.kind == 'STABLE' else "Unstable"
            label = (f"{info['system']} {side} Manifold "
                     f"({info['equilibrium']})")
            obj = _new_object(context, label, verts, faces,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            me = obj.data
            if info.get('near_equilibrium'):
                self.report(
                    {'WARNING'},
                    f"the ring passed within 0.05 of another "
                    f"equilibrium on {info['near_equilibrium']} "
                    f"checkpoints, where the field nearly vanishes")
            if info.get('outer_points', 0) >= self.max_ring_points:
                self.report(
                    {'WARNING'},
                    f"the outermost ring hit the "
                    f"{self.max_ring_points}-point cap, so it is "
                    f"under-resolved -- raise Max Ring Points or "
                    f"lower Arclength")
            how = ("a spiral saddle (complex pair)" if info['spiral']
                   else "a real pair")
            self.report({'INFO'},
                        f"{label}: {info['rings']} rings, arclength "
                        f"{info['arclength']:.0f}, "
                        f"{len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces; eigenvalues "
                        f"{np.round(info['eigenvalues'], 3)} -- {how}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'system')
            names = SYSTEMS[self.system][3]
            lay.prop(self, 'p1', text=names[0])
            lay.prop(self, 'p2', text=names[1])
            lay.prop(self, 'p3', text=names[2])
            lay.prop(self, 'equilibrium')
            lay.prop(self, 'kind', expand=True)
            lay.separator()
            for k in ('arclength', 'ring_spacing', 'target_edge',
                      'seed_radius', 'seed_points', 'max_ring_points',
                      'step'):
                lay.prop(self, k)
            lay.separator()
            for k in ('thickness', 'scale', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.invariant_manifold_add",
                             icon='SURFACE_NSURFACE')

    _classes = (MESH_OT_invariant_manifold_add,)

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


def _closest_approach(field, p, S, sign=1.0, tmax=3.0, dt=0.002,
                      centre=None, bound=1.0e4):
    """How near the flow brings each point to `centre`, as a fraction
    of where it started.  sign = +1 flows forward, -1 backward.

    Escaping trajectories have to be handled explicitly.  Running the
    Lorenz field BACKWARD -- which is what tests an unstable manifold
    -- diverges to infinity in short order, and once a point overflows
    every later comparison against it is a silent pass, because
    anything compared with NaN is False.  So a point that leaves
    `bound` is frozen at its best-so-far and stops being integrated."""
    c = np.zeros(3) if centre is None else np.asarray(centre, float)
    X = np.asarray(S, dtype=float).copy()
    best = np.linalg.norm(X - c, axis=1).copy()
    alive = np.ones(len(X), dtype=bool)
    for _ in range(int(tmax / dt)):
        k1 = sign * field(X, p)
        k2 = sign * field(X + 0.5 * dt * k1, p)
        k3 = sign * field(X + 0.5 * dt * k2, p)
        k4 = sign * field(X + dt * k3, p)
        Y = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ok = (np.all(np.isfinite(Y), axis=1)
              & (np.linalg.norm(Y - c, axis=1) < bound))
        alive &= ok
        if not alive.any():
            break
        X = np.where(alive[:, None], Y, X)
        r = np.linalg.norm(X - c, axis=1)
        best = np.where(alive, np.minimum(best, r), best)
    return best / np.maximum(np.linalg.norm(S - c, axis=1), 1e-12)


def _selftest():
    LP = SYSTEMS['LORENZ'][4]

    # ---- 1. the numeric Jacobian is the analytic one ----------------
    rng = np.random.default_rng(0)
    for _ in range(6):
        X = rng.normal(size=3) * 5.0
        err = float(np.max(np.abs(numeric_jacobian(lorenz_field, X, LP)
                                  - lorenz_jacobian_exact(X, LP))))
        if err > 1e-6:
            raise AssertionError(f"numeric Jacobian off by {err:.2e} "
                                 f"at {np.round(X, 3)}")
    print("Jacobian: central differences match the analytic Lorenz "
          "matrix to 1e-6")

    # ---- 2. the equilibria and their eigenstructure ------------------
    eqs = lorenz_equilibria(LP)
    if [nm for nm, _ in eqs] != ["Origin", "C+", "C-"]:
        raise AssertionError(f"Lorenz equilibria {[n for n, _ in eqs]}")
    for nm, q in eqs:
        r = float(np.max(np.abs(lorenz_field(q, LP))))
        if r > 1e-9:
            raise AssertionError(f"{nm} is not an equilibrium: "
                                 f"|f| = {r:.2e}")
    _u1, _u2, sel = invariant_eigenbasis(lorenz_field, eqs[0][1], LP,
                                         'STABLE')
    got = sorted(float(v.real) for v in sel)
    if any(abs(a - b) > 1e-3
           for a, b in zip(got, sorted([-22.8277, -8.0 / 3.0]))):
        raise AssertionError(f"origin stable eigenvalues {got}")

    # the origin's UNSTABLE side is one-dimensional, and saying so is
    # more useful than growing nonsense
    try:
        invariant_eigenbasis(lorenz_field, eqs[0][1], LP, 'UNSTABLE')
    except ValueError as e:
        if 'one-dimensional' not in str(e):
            raise AssertionError(f"unhelpful refusal: {e}")
    else:
        raise AssertionError("the origin's unstable manifold is a "
                             "curve; that should have been refused")

    # C+ is a SPIRAL saddle: its 2-D unstable eigenspace comes from a
    # complex pair, which the Lorenz-only version refused outright
    u1, u2, sel = invariant_eigenbasis(lorenz_field, eqs[1][1], LP,
                                       'UNSTABLE')
    if np.max(np.abs(sel.imag)) < 1e-9:
        raise AssertionError("C+ should have a complex unstable pair")
    if abs(float(sel[0].real) - 0.094) > 1e-2:
        raise AssertionError(f"C+ unstable eigenvalues {sel}")
    for a, b, want in ((u1, u1, 1.0), (u2, u2, 1.0), (u1, u2, 0.0)):
        if abs(float(np.dot(a, b)) - want) > 1e-12:
            raise AssertionError("the eigenbasis is not orthonormal")
    J = numeric_jacobian(lorenz_field, eqs[1][1], LP)
    n = np.cross(u1, u2)
    for v in (u1, u2):
        if abs(float(np.dot(J @ v, n))) > 1e-6:
            raise AssertionError("the eigenbasis does not span an "
                                 "invariant plane")
    print(f"equilibria: origin (real pair; unstable side correctly "
          f"refused as 1-D) and C+ (spiral saddle, "
          f"{np.round(sel, 3)})")

    # ---- 3. what each case can actually prove -----------------------
    # The defining local property (Hadamard-Perron) is that the
    # manifold leaves the equilibrium TANGENT to the invariant
    # eigenplane.  That is checkable for every case, so it is the
    # general test.
    cases = (('LORENZ', 0, 'STABLE', 1.0),
             ('LORENZ', 1, 'UNSTABLE', -1.0),
             ('ROSSLER', 0, 'UNSTABLE', -1.0))
    for system, eqi, kind, flow in cases:
        field = SYSTEMS[system][1]
        p = SYSTEMS[system][4]
        centre = SYSTEMS[system][2](p)[eqi][1]
        u1, u2, _sel = invariant_eigenbasis(field, centre, p, kind)
        nrm = np.cross(u1, u2)
        nrm = nrm / np.linalg.norm(nrm)
        rings, info = grow_manifold(system=system, equilibrium=eqi,
                                    kind=kind, arclength=1.0,
                                    ring_spacing=0.25)
        r = rings[1]
        out = (float(np.max(np.abs((r - centre) @ nrm)))
               / float(np.max(np.linalg.norm(r - centre, axis=1))))
        if out > 2e-2:
            raise AssertionError(
                f"{system} {info['equilibrium']} {kind}: the surface "
                f"leaves the equilibrium {out:.2e} out of its "
                f"invariant eigenplane -- it is not tangent to it")

        # growth is NOT monotone in general: at a spiral saddle the
        # rotation dwarfs the growth -- C+ turns at 10.195 while
        # growing at 0.094, so the ring pulses as it winds.  Only the
        # overall trend has to rise.
        rings, info = grow_manifold(system=system, equilibrium=eqi,
                                    kind=kind, arclength=20.0)
        lengths = [float(np.linalg.norm(
            np.diff(np.vstack([q, q[:1]]), axis=0), axis=1).sum())
            for q in rings]
        third = max(1, len(lengths) // 3)
        if (lengths[-1] <= lengths[0]
                or np.mean(lengths[-third:]) <= np.mean(
                    lengths[:third])):
            raise AssertionError(
                f"{system} {kind}: the rings do not grow overall "
                f"({lengths[0]:.3f} -> {lengths[-1]:.3f})")
        print(f"{info['system']:8s} {info['equilibrium']:6s} "
              f"{kind:8s}: tangent to its eigenplane to {out:.1e}, "
              f"circumference {lengths[0]:.3f} -> {lengths[-1]:.1f}")

    # The stronger test -- flow the surface and watch it collapse into
    # the equilibrium, against a control of points nudged off it --
    # only DISCRIMINATES where the contraction beats the transverse
    # blow-up.  It does at the Lorenz origin (rates 2.67 and 22.83).
    # It cannot at C+: running backward to test an unstable manifold
    # amplifies the transverse direction at e^(13.855 t) while the
    # spiral contracts at only e^(0.094 t), so every trajectory leaves
    # the box long before the collapse would show.  Asserting a weak
    # threshold there would be theatre, so the test is applied where
    # it means something and skipped, explicitly, where it does not.
    field, p = SYSTEMS['LORENZ'][1], SYSTEMS['LORENZ'][4]
    rings, _ = grow_manifold(arclength=20.0)
    outer = rings[-1]
    S = outer[np.linspace(0, len(outer) - 1, 16).astype(int)]
    on = _closest_approach(field, p, S, sign=1.0)
    rng2 = np.random.default_rng(1)
    off = _closest_approach(field, p,
                            S + 0.5 * rng2.normal(size=S.shape),
                            sign=1.0)
    if not (np.all(np.isfinite(on)) and np.all(np.isfinite(off))):
        raise AssertionError(
            "the invariance measurement came out non-finite, which "
            "would compare False against every threshold and pass "
            "silently")
    if float(np.median(on)) > 0.12:
        raise AssertionError(
            f"the Lorenz stable manifold should dive into the origin, "
            f"but reached only {float(np.median(on)):.3f} of its "
            f"radius")
    if float(np.median(off)) < 4.0 * float(np.median(on)):
        raise AssertionError(
            f"points nudged 0.5 off reached {float(np.median(off)):.3f} "
            f"against the surface's {float(np.median(on)):.3f}; the "
            f"test does not discriminate")
    print(f"invariance (Lorenz origin, where the rates allow it): the "
          f"surface dives to {float(np.median(on)):.3f} of its radius, "
          f"points nudged 0.5 off only to {float(np.median(off)):.3f}")

    # ---- 4. the mesh ------------------------------------------------
    V, F, info = build_invariant_manifold(arclength=30.0)
    cnt = _edge_counts(F)
    boundary = [e for e, c in cnt.items() if c == 1]
    if [e for e, c in cnt.items() if c > 2]:
        raise AssertionError("edges with more than two faces; the band "
                             "stitching is wrong")
    chi = len(V) - len(cnt) + len(F)
    if chi != 1:
        raise AssertionError(f"a manifold grown from a point is a disk, "
                             f"so chi should be 1, got {chi}")
    if len(boundary) != info['outer_points']:
        raise AssertionError(
            f"{len(boundary)} boundary edges against an outer ring of "
            f"{info['outer_points']} points")
    ext = float((V.max(axis=0) - V.min(axis=0)).max())
    if abs(ext - 2.0) > 1e-6:
        raise AssertionError(f"{ext:.4f} across, expected a 2 m fit")
    print(f"mesh: chi {chi}, one rim of {len(boundary)} edges, "
          f"{len(V)} verts, {len(F)} faces")

    # ---- 5. cubic resampling still earns its place ------------------
    field, p = SYSTEMS['LORENZ'][1], SYSTEMS['LORENZ'][4]

    def dive(rs):
        o = rs[-1]
        return float(np.median(_closest_approach(
            field, p, o[np.linspace(0, len(o) - 1, 16).astype(int)])))

    da = dive(grow_manifold(arclength=30.0)[0])
    db = dive(grow_manifold(arclength=30.0, cubic=False)[0])
    if da >= db:
        raise AssertionError(f"cubic {da:.3f} should beat linear "
                             f"{db:.3f}")
    print(f"resampling: Catmull-Rom {da:.3f} vs linear {db:.3f} -- the "
          f"interpolation is the bottleneck, not the integrator")

    # ---- 6. the growth direction ------------------------------------
    r0, _ = grow_manifold(arclength=2.0, ring_spacing=2.0)
    if (float(np.mean(np.linalg.norm(r0[-1], axis=1)))
            <= float(np.mean(np.linalg.norm(r0[0], axis=1)))):
        raise AssertionError("the ring did not move outward; the "
                             "integration sign is flipped")
    print("RESULT: OK")
