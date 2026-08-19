"""Fairing: drive the surface towards a discrete minimal surface.

The relaxation in :mod:`seifert.relax` is the paper's own method -- a
particle model borrowed from KnotPlot, which the paper is explicit is a
*substitute* for computing a minimal surface, chosen because it runs at
interactive rates.  It rounds a knot out nicely but it is a poor smoother: the
forces are isotropic in the ambient space and know nothing about the surface, so
they flatten curvature only incidentally.

This module does the thing the substitute stands in for.  Implicit fairing
(Desbrun, Meyer, Schroder and Barr, "Implicit Fairing of Irregular Meshes using
Diffusion and Curvature Flow", SIGGRAPH 1999) integrates mean-curvature flow
with a backward Euler step,

    (I + lambda * L) X' = X

with ``L`` the (random-walk normalised) cotangent Laplacian.  Because the step
is implicit it is unconditionally stable, so a single solve with a large lambda
does what hundreds of explicit smoothing passes would, without the shrinkage and
without the stability limit.  Holding the boundary fixed makes the stationary
point of the flow a surface of zero mean curvature spanning that boundary -- a
discrete minimal surface.

The cotangent weights are Pinkall and Polthier's ("Computing Discrete Minimal
Surfaces and Their Conjugates", Experimental Mathematics 2(1), 1993).

Implementation note: the reference port solves the sparse system with SciPy's
``spsolve``.  Blender's bundled Python ships NumPy but not SciPy, so this port
keeps the identical matrix -- the row-normalised ``(1 + lambda) I - lambda P``
with ``P = D^-1 W`` -- and solves it with a damped Jacobi iteration instead.
The matrix is a diagonally dominant M-matrix (row-sum of the off-diagonal is
``lambda`` against a diagonal of ``1 + lambda``), so Jacobi converges
geometrically with factor ``lambda / (1 + lambda)``; the iteration below runs to
a relative residual tolerance rather than a fixed count so it stays accurate as
``strength`` grows.  The matvec is a weighted average of neighbours, evaluated
with ``numpy.add.at`` over the directed cotangent edges -- no SciPy required.

Two further fairing modes complete the stack (S6.2/S6.3 of the solver
survey), both taking the mollified cotan weights from ``solver/cotan``:

* :func:`biharmonic_fair` -- Botsch-Kobbelt k=2 (thin-plate) fairing, an
  EQUILIBRIUM solve of the bi-Laplacian system with the rim and its first
  ring locked.  Mean-curvature flow is a shrinker whose only brake is the
  pinned rim (measured here: an open cylinder's waist walks to the
  catenoid's 0.848 R as iterations grow); the bi-harmonic solve holds the
  same waist at 0.998 R, because shrink is not a solution of its equation.
  Botsch and Kobbelt, "An Intuitive Framework for Real-Time Freeform
  Modeling", ACM Trans. Graph. 23(3) (2004).

* :func:`cmcf_fair` -- conformalized MCF: the stiffness matrix is FROZEN
  at its initial-surface value while the mass matrix tracks the current
  surface, removing the weight-degradation feedback that makes long MCF
  runs pinch.  Kazhdan, Solomon, Ben-Chen, "Can Mean-Curvature Flow Be
  Made Non-Singular?", Computer Graphics Forum 31(5) (2012).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh

try:
    from ..solver import cotan as _solver_cotan
except ImportError:                      # flat (path-based) headless import
    try:
        from solver import cotan as _solver_cotan
    except ImportError:
        _solver_cotan = None

__all__ = ["cotangent_laplacian", "minimal_surface", "smooth_boundary",
           "biharmonic_fair", "cmcf_fair"]


@dataclass(frozen=True)
class _CotangentLaplacian:
    """Row-normalised cotangent Laplacian ``L = I - D^-1 W`` in edge form.

    Stored as directed edges ``src -> dst`` carrying symmetric weights ``w`` and
    the per-vertex weight sum ``degree``, which is all the Jacobi solve and the
    matvec need.  Duplicate ``(src, dst)`` pairs -- one per adjacent triangle of
    an interior edge -- are summed by :func:`numpy.add.at`, exactly as the sparse
    assembly summed the duplicate COO entries.
    """

    n: int
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    degree: np.ndarray

    def averaged(self, x: np.ndarray) -> np.ndarray:
        """The neighbour average ``(D^-1 W) x`` -- one sparse matvec."""
        acc = np.zeros_like(x)
        np.add.at(acc, self.src, self.weight[:, None] * x[self.dst])
        return acc / self.degree[:, None]


def cotangent_laplacian(mesh: Mesh, clamp: float = 1e-3) -> _CotangentLaplacian:
    """Row-normalised cotangent Laplacian of ``mesh`` in directed-edge form.

    Weights are clamped below at ``clamp`` so a badly shaped triangle cannot make
    the system indefinite; on the meshes this package builds the clamp fires on
    well under a percent of edges.
    """
    tri = mesh.triangulated()
    P = tri.vertices
    n = len(P)
    faces = np.asarray(tri.faces, dtype=int)
    if not len(faces):
        empty = np.zeros(0, dtype=int)
        return _CotangentLaplacian(n, empty, empty, np.zeros(0),
                                   np.ones(max(n, 1)))

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        c = 3 - a - b
        i, j, k = faces[:, a], faces[:, b], faces[:, c]
        u = P[i] - P[k]
        v = P[j] - P[k]
        cross = np.linalg.norm(np.cross(u, v), axis=1)
        cross[cross < 1e-14] = 1e-14
        cot = np.einsum("ij,ij->i", u, v) / cross
        w = np.maximum(0.5 * cot, clamp)
        # symmetric weight: both directions of the edge (i, j)
        src_parts += [i, j]
        dst_parts += [j, i]
        w_parts += [w, w]

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    weight = np.concatenate(w_parts)
    degree = np.zeros(n)
    np.add.at(degree, src, weight)
    degree[degree == 0] = 1.0
    return _CotangentLaplacian(n, src, dst, weight, degree)


def _implicit_solve(
    laplacian: _CotangentLaplacian,
    rhs: np.ndarray,
    strength: float,
    pinned: np.ndarray | None,
    tol: float = 1e-6,
    max_iterations: int = 4000,
) -> np.ndarray:
    """Solve ``(I + strength * L) X = rhs`` by damped Jacobi, rim pinned.

    With ``L = I - P`` the update is
    ``X <- (rhs + strength * P X) / (1 + strength)``, and pinned rows are held at
    their right-hand side.  Convergence factor ``strength / (1 + strength) < 1``,
    so the loop is unconditionally stable for any positive ``strength``.
    """
    x = rhs.copy()
    denom = 1.0 + strength
    scale = max(1.0, float(np.linalg.norm(rhs)))
    for _ in range(max_iterations):
        nxt = (rhs + strength * laplacian.averaged(x)) / denom
        if pinned is not None and len(pinned):
            nxt[pinned] = rhs[pinned]
        delta = float(np.linalg.norm(nxt - x))
        x = nxt
        if delta <= tol * scale:
            break
    return x


def _implicit_solve_cg(
    V: np.ndarray,
    T: np.ndarray,
    rhs: np.ndarray,
    strength: float,
    pinned: np.ndarray | None,
    tol: float = 1e-8,
    max_iterations: int = 4000,
) -> np.ndarray:
    """Solve the same implicit-fairing system in its symmetric form
    ``(D + strength * (D - W)) X = D X0`` by Jacobi-preconditioned CG,
    with RAW (mollified-length) cotangent weights -- no positivity
    floor.  ``D - W`` is the standard cotan Laplacian and is PSD for any
    real triangulation regardless of the signs of individual weights
    (it is the Hessian of the Dirichlet energy), so CG applies where
    the damped-Jacobi argument (diagonal dominance) does not.  ``D`` is
    floored at a tiny positive value in the mass term only, guarding
    the rare pathological vertex star with a negative weight sum."""
    E, W = _solver_cotan.edge_cotan_weights(V, T, mode="mollify")
    n = len(V)
    deg = np.zeros(n)
    np.add.at(deg, E[:, 0], W)
    np.add.at(deg, E[:, 1], W)
    scale = float(np.mean(np.abs(deg))) or 1.0
    D = np.maximum(deg, 1e-9 * scale)

    free = np.ones(n, dtype=bool)
    if pinned is not None and len(pinned):
        free[pinned] = False

    def matvec_full(X):
        y = (1.0 + strength) * D[:, None] * X
        np.add.at(y, E[:, 0], -strength * W[:, None] * X[E[:, 1]])
        np.add.at(y, E[:, 1], -strength * W[:, None] * X[E[:, 0]])
        return y

    Xb = np.where(~free[:, None], rhs, 0.0)
    b = (D[:, None] * rhs - matvec_full(Xb))[free]

    def matvec(xf):
        full = np.zeros((n, 3))
        full[free] = xf
        return matvec_full(full)[free]

    Minv = 1.0 / ((1.0 + strength) * D[free])[:, None]
    x = rhs[free].copy()
    r = b - matvec(x)
    z = Minv * r
    p = z.copy()
    rz = np.sum(r * z, axis=0)
    b_norm = max(float(np.max(np.sum(b * b, axis=0))), 1e-300)
    for _ in range(max_iterations):
        Ap = matvec(p)
        pAp = np.sum(p * Ap, axis=0)
        alpha = rz / np.where(np.abs(pAp) > 1e-300, pAp, 1e-300)
        x += alpha * p
        r -= alpha * Ap
        if float(np.max(np.sum(r * r, axis=0))) < tol * tol * b_norm:
            break
        z = Minv * r
        rz_new = np.sum(r * z, axis=0)
        p = z + (rz_new / np.maximum(rz, 1e-300)) * p
        rz = rz_new
    out = rhs.copy()
    out[free] = x
    return out


def minimal_surface(
    mesh: Mesh,
    strength: float = 12.0,
    iterations: int = 4,
    fix_boundary: bool = True,
    mollify: bool = True,
) -> Mesh:
    """Flow the surface towards zero mean curvature with its rim held fixed.

    ``strength`` is the implicit time step: larger means a bigger jump towards
    the minimal surface per solve.  A handful of iterations is plenty, since the
    Laplacian is recomputed each time and the flow converges quickly.

    ``mollify=True`` swaps the floored-weight damped-Jacobi solve for
    the symmetric raw-cotangent system (intrinsic mollification, Sharp-
    Crane 2020) solved by preconditioned CG: the historical floor fires
    on EVERY obtuse corner (~20% of corners on these surfaces, measured
    in tests/bench), biasing the stationary point away from the true
    discrete minimal surface.

    >>> from seifert.build import seifert_surface
    >>> m = minimal_surface(seifert_surface("AAA"), iterations=2)
    >>> m.info().genus
    1
    """
    positions = mesh.vertices.copy()
    boundary = np.array(
        sorted({v for loop in mesh.boundary_loops() for v in loop}), dtype=int
    )
    pinned = boundary if (fix_boundary and len(boundary)) else None
    use_cg = mollify and _solver_cotan is not None
    T = None
    if use_cg:
        tri = mesh.triangulated()
        T = np.asarray(tri.faces, dtype=np.int64)
    for _ in range(iterations):
        rhs = positions.copy()
        if use_cg:
            positions = _implicit_solve_cg(positions, T, rhs, strength,
                                           pinned)
        else:
            work = Mesh(positions, mesh.faces)
            laplacian = cotangent_laplacian(work)
            positions = _implicit_solve(laplacian, rhs, strength, pinned)
    return Mesh(positions, list(mesh.faces), list(mesh.face_groups))


def _tri_arrays(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    tri = mesh.triangulated()
    return tri.vertices.copy(), np.asarray(tri.faces, dtype=np.int64)


def _vertex_areas(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Barycentric (star/3) vertex areas -- the lumped mass matrix."""
    fa = 0.5 * np.linalg.norm(
        np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    a = np.zeros(len(V))
    for k in range(3):
        np.add.at(a, T[:, k], fa / 3.0)
    return np.maximum(a, 1e-300)


def _stiffness(V: np.ndarray, T: np.ndarray):
    """Mollified cotan stiffness in half-edge form: (E, W, deg) with
    L x = deg * x - sum_j w_ij x_j (PSD for any real triangulation)."""
    if _solver_cotan is None:
        raise RuntimeError("solver.cotan is unavailable; biharmonic and "
                           "cMCF fairing need the shared solver core")
    E, W = _solver_cotan.edge_cotan_weights(V, T, mode="mollify")
    deg = np.zeros(len(V))
    np.add.at(deg, E[:, 0], W)
    np.add.at(deg, E[:, 1], W)
    return E, W, deg


def _apply_L(E, W, deg, X):
    Y = deg[:, None] * X
    n = len(Y)
    for c in range(3):
        xc = X[:, c]
        Y[:, c] -= np.bincount(E[:, 0], W * xc[E[:, 1]], minlength=n)
        Y[:, c] -= np.bincount(E[:, 1], W * xc[E[:, 0]], minlength=n)
    return Y


def _pcg(matvec, b, Minv, x0=None, tol=1e-9, max_iterations=4000):
    """Jacobi-preconditioned CG on the free rows of an SPD system,
    columns solved together as (n, 3).  b and the returned x are
    free-row arrays."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - matvec(x)
    z = Minv[:, None] * r
    p = z.copy()
    rz = float(np.einsum('ij,ij->', r, z))
    b_norm2 = max(float(np.einsum('ij,ij->', b, b)), 1e-300)
    for _ in range(max_iterations):
        Ap = matvec(p)
        pAp = float(np.einsum('ij,ij->', p, Ap))
        if not (pAp > 0.0):
            break
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        if float(np.einsum('ij,ij->', r, r)) < tol * tol * b_norm2:
            break
        z = Minv[:, None] * r
        rz_new = float(np.einsum('ij,ij->', r, z))
        p = z + (rz_new / max(rz, 1e-300)) * p
        rz = rz_new
    return x


def _boundary_rings(mesh: Mesh, T: np.ndarray, rings: int = 1) -> np.ndarray:
    """Boundary vertices plus `rings` rings of their neighbours, as a
    boolean pin mask over the triangulated vertex array."""
    n = len(mesh.vertices)
    pinned = np.zeros(n, dtype=bool)
    b = [v for loop in mesh.boundary_loops() for v in loop]
    pinned[b] = True
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    for _ in range(rings):
        grow = pinned.copy()
        sel = pinned[E[:, 0]]
        grow[E[sel, 1]] = True
        sel = pinned[E[:, 1]]
        grow[E[sel, 0]] = True
        pinned = grow
    return pinned


def biharmonic_fair(mesh: Mesh, iterations: int = 1,
                    tol: float = 1e-9) -> Mesh:
    """Botsch-Kobbelt k=2 (thin-plate) fairing with the rim pinned:
    solve the bi-Laplacian system  L M^-1 L X = 0  for the interior,
    with the boundary AND its first ring locked at their current
    positions (the k-1 = 1 ring of Dirichlet data that gives tangent
    continuity at the rim, which position-only pinning never does).

    This is an EQUILIBRIUM solve, not a flow: the mean-curvature-flow
    shrink bias of :func:`minimal_surface` -- a shrinker whose only
    brake is the pinned rim -- cannot exist here by construction.  The
    interior lands on the discrete thin-plate span of the rim data in
    one solve; `iterations` > 1 re-derives the (geometry-dependent)
    cotan weights from each new surface and re-solves, a fixed-point
    polish whose movement drops sharply after the first pass.

    L is the mollified-cotan stiffness (solver/cotan), M the lumped
    (barycentric) mass; the system is SPD on the free rows and solved
    by Jacobi-preconditioned CG (the bi-Laplacian squares the
    condition number, so plain damped Jacobi is not an option).

    References: M. Botsch and L. Kobbelt, "An Intuitive Framework for
    Real-Time Freeform Modeling", ACM Trans. Graph. 23(3) (2004);
    the pmp-library fairing implementation is the reference spec."""
    V, T = _tri_arrays(mesh)
    if not len(T):
        return Mesh(V, list(mesh.faces), list(mesh.face_groups))
    pinned = _boundary_rings(mesh, T, rings=1)
    free = ~pinned
    if not free.any():
        return Mesh(V, list(mesh.faces), list(mesh.face_groups))
    for _ in range(max(1, iterations)):
        E, W, deg = _stiffness(V, T)
        Minv_mass = 1.0 / _vertex_areas(V, T)

        def K_full(X):
            return _apply_L(E, W, deg, Minv_mass[:, None]
                            * _apply_L(E, W, deg, X))

        # Dirichlet reduction: b = -(K X_pinned) on the free rows
        Xb = np.where(pinned[:, None], V, 0.0)
        b = -K_full(Xb)[free]

        def K_free(Xf):
            X = np.zeros_like(V)
            X[free] = Xf
            return K_full(X)[free]

        # Jacobi diagonal of K = L^T diag(Minv) L:
        # diag_v = Minv_v deg_v^2 + sum_{j~v} Minv_j w_vj^2
        diagK = Minv_mass * deg * deg
        np.add.at(diagK, E[:, 0], Minv_mass[E[:, 1]] * W * W)
        np.add.at(diagK, E[:, 1], Minv_mass[E[:, 0]] * W * W)
        Minv_pc = 1.0 / np.maximum(diagK[free], 1e-300)
        V[free] = _pcg(K_free, b, Minv_pc, x0=V[free], tol=tol)
    return Mesh(V, list(mesh.faces), list(mesh.face_groups))


def cmcf_fair(mesh: Mesh, strength: float = 2.0, iterations: int = 10,
              fix_boundary: bool = True, tol: float = 1e-9) -> Mesh:
    """Conformalized mean-curvature flow (Kazhdan-Solomon-Ben-Chen):
    implicit MCF steps in which the cotan stiffness L0 is FROZEN at its
    initial-surface value while the lumped mass matrix is re-derived
    from the current surface each step,

        (M_t + delta * L0) X' = M_t X,       delta = strength * mean(M_t),

    (normalizing the mass by its mean is exactly an adaptive time step,
    which is how the reference implementation keeps the step meaningful
    as the surface shrinks).  Freezing L0 removes the feedback loop
    that makes ordinary MCF pinch necks and blow up cotan weights on
    long runs: the weights can never degrade, because they are never
    recomputed.  The price is the stationary state: iterating drives
    the surface to the L0-HARMONIC span of its rim (the first
    Pinkall-Polthier iterate, held in the initial metric), not to the
    zero-mean-curvature surface -- for a closed surface the flow
    conformalizes it toward a round sphere, the paper's headline.

    Reference: M. Kazhdan, J. Solomon, M. Ben-Chen, "Can Mean-Curvature
    Flow Be Made Non-Singular?", Computer Graphics Forum 31(5) (2012)."""
    V, T = _tri_arrays(mesh)
    if not len(T):
        return Mesh(V, list(mesh.faces), list(mesh.face_groups))
    boundary = np.array(
        sorted({v for loop in mesh.boundary_loops() for v in loop}),
        dtype=int)
    pinned = np.zeros(len(V), dtype=bool)
    if fix_boundary and len(boundary):
        pinned[boundary] = True
    free = ~pinned
    E0, W0, deg0 = _stiffness(V, T)      # frozen at the initial surface
    for _ in range(max(1, iterations)):
        a = _vertex_areas(V, T)
        delta = float(strength) * float(np.mean(a))

        def A_full(X):
            return a[:, None] * X + delta * _apply_L(E0, W0, deg0, X)

        rhs = a[:, None] * V
        Xb = np.where(pinned[:, None], V, 0.0)
        b = (rhs - A_full(Xb))[free]

        def A_free(Xf):
            X = np.zeros_like(V)
            X[free] = Xf
            return A_full(X)[free]

        Minv_pc = 1.0 / np.maximum((a + delta * deg0)[free], 1e-300)
        V[free] = _pcg(A_free, b, Minv_pc, x0=V[free], tol=tol)
    return Mesh(V, list(mesh.faces), list(mesh.face_groups))


def smooth_boundary(mesh: Mesh, iterations: int = 20, weight: float = 0.5) -> Mesh:
    """Fair the boundary curves in place.

    The rim of a freshly built surface is a chain of disk-rim arcs and band
    edges meeting at corners.  A Laplacian pass along each closed loop removes
    the corners without moving the curve off its knot type.
    """
    positions = mesh.vertices.copy()
    loops = [np.asarray(loop, dtype=int) for loop in mesh.boundary_loops()]
    for _ in range(iterations):
        for loop in loops:
            p = positions[loop]
            target = 0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))
            positions[loop] = p + weight * (target - p)
    return Mesh(positions, list(mesh.faces), list(mesh.face_groups))


def _cylinder_lathe(nring=32, nrow=13, h=0.5, R=1.0):
    """Open cylinder test mesh (two pinned rims) as a Mesh of triangles."""
    th = np.linspace(0.0, 2.0 * np.pi, nring, endpoint=False)
    zs = np.linspace(-h, h, nrow)
    V = np.array([[R * np.cos(t), R * np.sin(t), z]
                  for z in zs for t in th])
    faces = []
    for j in range(nrow - 1):
        for i in range(nring):
            a = j * nring + i
            b = j * nring + (i + 1) % nring
            faces.append((a, b, a + nring))
            faces.append((b, b + nring, a + nring))
    return Mesh(V, faces)


def _selftest():
    ok = True

    # --- bi-harmonic: linear data is reproduced exactly (a tilted
    # plane patch with a bumped interior returns to the plane) ---------
    n = 11
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n),
                         indexing='ij')
    V = np.stack([xs.ravel(), ys.ravel(),
                  (0.3 * xs + 0.2 * ys).ravel()], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.append((a, a + n, a + n + 1))
            faces.append((a, a + n + 1, a + 1))
    plane = Mesh(V.copy(), faces)
    rng = np.random.default_rng(2)
    bump = plane.vertices.copy()
    interior = np.ones(len(bump), dtype=bool)
    tri = plane.triangulated()
    T = np.asarray(tri.faces, dtype=np.int64)
    pinned = _boundary_rings(plane, T, rings=1)
    bump[~pinned] += 0.05 * rng.normal(size=(int((~pinned).sum()), 3))
    bumped = Mesh(bump, faces)
    flat = biharmonic_fair(bumped, iterations=1, tol=1e-12)
    resid = float(np.max(np.abs(flat.vertices[:, 2]
                                - (0.3 * flat.vertices[:, 0]
                                   + 0.2 * flat.vertices[:, 1]))))
    good = resid < 1e-8
    ok &= good
    print(f"fair: biharmonic reproduces the tilted plane "
          f"(max dev {resid:.1e}) {'OK' if good else 'FAIL'}")

    # --- shrinkage A/B on the open cylinder: MCF fairing is a shrinker
    # (waist -> catenoid ~0.85 R); the bi-harmonic equilibrium solve
    # must hold the waist essentially at the cylinder ------------------
    nring, nrow = 32, 13
    cyl = _cylinder_lathe(nring, nrow)
    mid = (nrow // 2) * nring

    def waist(m):
        return float(np.mean(np.hypot(m.vertices[mid:mid + nring, 0],
                                      m.vertices[mid:mid + nring, 1])))

    mcf = minimal_surface(cyl, strength=12.0, iterations=4)
    mcf_long = minimal_surface(cyl, strength=12.0, iterations=20)
    bih = biharmonic_fair(cyl, iterations=1)
    w_mcf, w_mcfl, w_bih = waist(mcf), waist(mcf_long), waist(bih)
    shrink_mcf = 1.0 - mcf.area() / cyl.area()
    shrink_bih = 1.0 - bih.area() / cyl.area()
    # MCF is a shrinker: more iterations, more waist loss (toward the
    # catenoid 0.848); the bi-harmonic solve is an equilibrium and
    # holds the cylinder to a fraction of a percent in ONE solve.
    good = (w_mcfl < w_mcf < 0.95 and w_bih > 0.97
            and shrink_bih < 0.3 * shrink_mcf)
    ok &= good
    print(f"fair: cylinder waist MCF {w_mcf:.4f} -> {w_mcfl:.4f} at 5x "
          f"budget (catenoid 0.848) vs biharmonic {w_bih:.4f}; area "
          f"shrink {shrink_mcf:.3f} vs {shrink_bih:.4f} "
          f"{'OK' if good else 'FAIL'}")

    # --- cMCF on the cylinder: frozen-Laplacian steps converge to the
    # L0-harmonic span -- a waist BETWEEN the catenoid and the cylinder
    # (mode-1 harmonic profile ~ 1/cosh(h/R) at the waist), rims fixed -
    cm = cmcf_fair(cyl, strength=4.0, iterations=40)
    w_cm = waist(cm)
    rim_move = float(np.max(np.linalg.norm(
        cm.vertices[:nring] - cyl.vertices[:nring], axis=1)))
    good = 0.85 < w_cm < 0.97 and rim_move == 0.0
    ok &= good
    print(f"fair: cMCF cylinder waist {w_cm:.4f} (harmonic span; "
          f"catenoid 0.848 < w < 1), rim move {rim_move:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # --- cMCF on a closed surface conformalizes toward the sphere -----
    try:
        from ..surfaces.primitives import icosphere
    except ImportError:
        from surfaces.primitives import icosphere    # type: ignore
    SV, SF = icosphere(2, 'per_level')
    SV = np.asarray(SV, float) * np.array([1.4, 0.75, 1.0])
    squash = Mesh(SV, [tuple(f) for f in np.asarray(SF)])
    r0 = np.linalg.norm(SV - SV.mean(axis=0), axis=1)
    cv0 = float(np.std(r0) / np.mean(r0))
    ball = cmcf_fair(squash, strength=2.0, iterations=25,
                     fix_boundary=False)
    r1 = np.linalg.norm(ball.vertices - ball.vertices.mean(axis=0), axis=1)
    cv1 = float(np.std(r1) / np.mean(r1))
    good = cv1 < 0.15 * cv0
    ok &= good
    print(f"fair: cMCF squashed sphere radius cv {cv0:.3f} -> {cv1:.4f} "
          f"{'OK' if good else 'FAIL'}")

    # --- the new modes leave the default path alone: minimal_surface
    # on a seifert surface still returns genus 1 for AAA ---------------
    try:
        from .build import seifert_surface
    except ImportError:
        from seifert.build import seifert_surface    # type: ignore
    m = seifert_surface("AAA")
    fair_m = minimal_surface(m, iterations=2)
    bi_m = biharmonic_fair(m)
    cm_m = cmcf_fair(m, iterations=4)
    good = (fair_m.info().genus == 1 and bi_m.info().genus == 1
            and cm_m.info().genus == 1)
    ok &= good
    print(f"fair: AAA genus preserved by all three fairings "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("seifert.fair self-test failed")
