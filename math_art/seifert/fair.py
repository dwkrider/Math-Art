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

__all__ = ["cotangent_laplacian", "minimal_surface", "smooth_boundary"]


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
