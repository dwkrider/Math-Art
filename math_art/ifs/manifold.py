# Invariant manifolds of continuous dynamical systems.
#
# Part of the Math Art ifs engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# The stable and unstable manifolds of a saddle equilibrium: the sets of
# initial conditions whose forward or backward orbits converge to it.
# Grown outward from the linear eigenspace by arclength-controlled
# continuation.  Sits beside `flow.py`; both integrate ODEs.
#
# References:
# - B. Krauskopf and H. Osinga, "Computing invariant manifolds via the
#   continuation of orbit segments", and the geodesic-level-set method,
#   SIAM Journal on Applied Dynamical Systems, 2003-2005.

import math
import numpy as np


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


_EQ_CACHE = {}


def find_equilibria(field, p, extent=30.0, grid=5, tol=1e-9,
                    max_iter=80):
    """Solve f = 0 by Newton from a fixed grid of seeds.

    Deterministic on purpose: the equilibrium dropdown indexes into
    this list, so a random seeding that reordered the roots between
    runs would silently change what the operator builds.  The seed
    grid is fixed, duplicates are merged by rounding, and the survivors
    are sorted by (norm, x, y, z)."""
    # the enum callback that lists equilibria fires on every redraw of
    # the redo panel, and Thomas needs 125 Newton solves; without this
    # the panel would re-solve them all on each mouse move
    ck = (getattr(field, '__name__', repr(field)),
          tuple(round(float(v), 9) for v in p), extent, grid)
    if ck in _EQ_CACHE:
        return _EQ_CACHE[ck]
    seeds = []
    span = np.linspace(-extent, extent, int(grid))
    for a in span:
        for b in span:
            for c in span:
                seeds.append((a, b, c))
    found = []
    for x0 in seeds:
        x = np.asarray(x0, dtype=float)
        for _ in range(int(max_iter)):
            fx = field(x, p)
            if not np.all(np.isfinite(fx)):
                break
            if float(np.max(np.abs(fx))) < tol:
                break
            J = numeric_jacobian(field, x, p)
            try:
                dx = np.linalg.lstsq(J, -fx, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(dx)):
                break
            step = float(np.linalg.norm(dx))
            if step > 10.0 * extent:          # diverging: give up
                break
            x = x + dx
        else:
            continue
        fx = field(x, p)
        if (np.all(np.isfinite(x))
                and float(np.max(np.abs(fx))) < tol
                and float(np.max(np.abs(x))) <= 4.0 * extent):
            found.append(x + 0.0)             # kill -0.0
    if not found:
        _EQ_CACHE[ck] = []
        return []
    # merge duplicates on a rounded key but keep the UNROUNDED root:
    # storing the rounded one would move it off the solution and leave
    # a residual around 1e-7 instead of 1e-12
    seen = {}
    for x in found:
        key = tuple(np.round(x, 5) + 0.0)
        seen.setdefault(key, x)
    roots = list(seen.values())
    roots.sort(key=lambda v: (round(float(np.linalg.norm(v)), 5),
                              round(float(v[0]), 5),
                              round(float(v[1]), 5),
                              round(float(v[2]), 5)))
    out = [(f"E{k}", v) for k, v in enumerate(roots)]
    _EQ_CACHE[ck] = out
    return out


def _lorenz_like_equilibria(p, zc, xc2):
    """The origin plus a symmetric pair, the shape shared by Lorenz,
    Chen, Rayleigh-Benard and their relatives: y = x, z = zc and
    x^2 = xc2."""
    out = [("Origin", np.zeros(3))]
    if xc2 > 0.0:
        r = math.sqrt(xc2)
        out.append(("C+", np.array([r, r, zc])))
        out.append(("C-", np.array([-r, -r, zc])))
    return out


def chen_field(X, p):
    """Chen-Celikovsky.  p = (a, b, c)."""
    a, b, c = p
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([a * (y - x), -x * z + c * y, x * y - b * z],
                    axis=-1)


def chen_equilibria(p):
    a, b, c = p
    return _lorenz_like_equilibria(p, c, b * c)


def rb_field(X, p):
    """Rayleigh-Benard convection.  p = (a, r, b)."""
    a, r, b = p
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([-a * (x - y), r * x - y - x * z, x * y - b * z],
                    axis=-1)


def rb_equilibria(p):
    a, r, b = p
    return _lorenz_like_equilibria(p, r - 1.0, b * (r - 1.0))


def shimizu_field(X, p):
    """Shimizu-Morioka.  p = (a, b, unused)."""
    a, b = p[0], p[1]
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([y, x - a * y - x * z, -b * z + x * x], axis=-1)


def shimizu_equilibria(p):
    """y = 0 throughout; then x(1 - z) = 0 gives the origin, or z = 1
    with x^2 = b."""
    a, b = p[0], p[1]
    out = [("Origin", np.zeros(3))]
    if b > 0.0:
        r = math.sqrt(b)
        out.append(("C+", np.array([r, 0.0, 1.0])))
        out.append(("C-", np.array([-r, 0.0, 1.0])))
    return out


def halvorsen_field(X, p):
    """Halvorsen's cyclically symmetric attractor.  p = (a, .., ..)."""
    a = p[0]
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([-a * x - 4.0 * y - 4.0 * z - y * y,
                     -a * y - 4.0 * z - 4.0 * x - z * z,
                     -a * z - 4.0 * x - 4.0 * y - x * x], axis=-1)


def halvorsen_equilibria(p):
    """No tidy closed form, so these are solved for."""
    return find_equilibria(halvorsen_field, p, extent=12.0, grid=5)


def thomas_field(X, p):
    """Thomas' cyclically symmetric attractor.  p = (b, .., ..)."""
    b = p[0]
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([-b * x + np.sin(y),
                     -b * y + np.sin(z),
                     -b * z + np.sin(x)], axis=-1)


def thomas_equilibria(p):
    """Transcendental, and there are many; solved for numerically."""
    return find_equilibria(thomas_field, p, extent=8.0, grid=5)


SYSTEMS = {
    'LORENZ': ("Lorenz", lorenz_field, lorenz_equilibria,
               ("Sigma", "Rho", "Beta"), (10.0, 28.0, 8.0 / 3.0)),
    'CHEN': ("Chen-Celikovsky", chen_field, chen_equilibria,
             ("a", "b", "c"), (36.0, 3.0, 20.0)),
    'RAYLEIGH': ("Rayleigh-Benard", rb_field, rb_equilibria,
                 ("a", "r", "b"), (9.0, 12.0, 5.0)),
    'SHIMIZU': ("Shimizu-Morioka", shimizu_field, shimizu_equilibria,
                ("a", "b", "(unused)"), (0.75, 0.45, 0.0)),
    'ROSSLER': ("Roessler", rossler_field, rossler_equilibria,
                ("a", "b", "c"), (0.2, 0.2, 5.7)),
    'HALVORSEN': ("Halvorsen", halvorsen_field, halvorsen_equilibria,
                  ("a", "(unused)", "(unused)"), (1.4, 0.0, 0.0)),
    'THOMAS': ("Thomas", thomas_field, thomas_equilibria,
               ("b", "(unused)", "(unused)"), (0.19, 0.0, 0.0)),
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


HYPERBOLIC_TOL = 1e-4


def classify_equilibrium(field, point, p, tol=HYPERBOLIC_TOL):
    """(eigenvalues, sides) where `sides` lists the kinds whose
    manifold is a SURFACE -- exactly two eigenvalues of that sign.

    An eigenvalue counts only if its real part is a real part.  At
    Roessler's outer equilibrium the complex pair has
    Re = -4.6e-6 against |lambda| = 5.43, a ratio of 8.5e-7: that
    equilibrium is not hyperbolic to within anything one can compute,
    the stable manifold theorem does not apply to it, and whether the
    pair reads as stable or unstable is decided by rounding.  Treating
    it as a stable manifold would be inventing a theorem."""
    w = np.linalg.eig(numeric_jacobian(field, point, p))[0]
    scale = float(np.max(np.abs(w)))
    if scale <= 0.0:
        return w, []
    hyper = np.abs(w.real) > tol * scale
    sides = []
    if int(np.sum(hyper & (w.real < 0.0))) == 2:
        sides.append('STABLE')
    if int(np.sum(hyper & (w.real > 0.0))) == 2:
        sides.append('UNSTABLE')
    return w, sides


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
    scale = max(float(np.max(np.abs(w))), 1e-300)
    side = "stable" if kind == 'STABLE' else "unstable"
    soft = np.abs(w.real) <= HYPERBOLIC_TOL * scale
    if np.any(soft):
        raise ValueError(
            f"this equilibrium is not hyperbolic -- "
            f"{int(np.sum(soft))} of its eigenvalues have a real part "
            f"of essentially zero "
            f"(|Re| / |lambda| = "
            f"{float(np.min(np.abs(w.real)) / scale):.1e}), so the "
            f"stable manifold theorem does not apply and there is no "
            f"{side} manifold to grow")
    want = (w.real < 0.0) if kind == 'STABLE' else (w.real > 0.0)
    idx = np.nonzero(want)[0]
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
