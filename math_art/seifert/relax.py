"""Physically-based relaxation, after section V.C of van Wijk and Cohen.

The model is borrowed from Scharein's KnotPlot: neighbouring vertices attract,
distant ones repel, and the whole thing is integrated explicitly with damping
and a time step that decays geometrically so a run settles instead of ringing.

    F_a(r) = H * r ** (1 + beta)        between neighbours
    F_r(r) = K * r ** -(2 + alpha)      between everything else

    v <- (1 - gamma) * v + F * dt
    p <- p + min(d_max, |v dt|) * unit(v dt)
    dt <- (1 - mu) * dt

The division of labour is what makes this produce a taut surface rather than a
collapsed one.  The *link* -- the boundary of the surface -- gets both forces,
so it spreads itself out and stays embedded.  The *interior* of the surface gets
attraction only, so with its rim held out by the link it pulls itself tight: a
discrete tension minimiser, which is the paper's substitute for computing a
genuine minimal surface.  Quoting the paper, the surface "follows the link, but
does not influence it".

Radii are normalised by the initial mean edge length, so the parameters mean the
same thing whatever scale the input is at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .mesh import Mesh

__all__ = ["RelaxParams", "relax"]


@dataclass(frozen=True)
class RelaxParams:
    """Force-model and integrator settings.

    The paper reports alpha = 0, beta = 1, and attraction = repulsion (so H = K),
    run for many thousands of iterations.  The remaining constants are
    unpublished and chosen here to be stable for the meshes this package builds.
    """

    attraction: float = 1.0          # H  (= repulsion K)
    repulsion: float = 0.4           # K
    beta: float = 1.0                # attraction exponent offset (paper: 1)
    alpha: float = 0.0               # repulsion exponent offset  (paper: 0)
    damping: float = 0.15            # gamma
    #: mu.  The reference method runs many thousands of iterations, so its decay
    #: must be far gentler than a few-hundred-step run wants; this is set so
    #: that dt has fallen by about half after 1,000 steps.
    step_decay: float = 0.0007
    time_step: float = 0.25          # initial dt
    max_move: float = 0.08           # d_max, as a multiple of the mean edge length
    #: d_close, likewise.  Scharein's collision guard: a move that would bring a
    #: link vertex nearer than this to a non-neighbouring segment is refused.
    #: Set it too low and the knot quietly passes through itself and flattens --
    #: the mesh stays combinatorially correct, but the *embedding* untangles.
    #: For a trefoil, 0.35 relaxes to writhe about -0.4; 1.5 holds it at -3.1.
    #: 0 disables the check.
    min_clearance: float = 1.5
    pin_boundary: bool = False       # freeze the link and relax only the interior
    #: scales the interior force only.  At 1.0 the surface pulls itself fully
    #: taut, which is the minimal surface; lower values let the surface follow
    #: the link while keeping more of the disk-and-band shape it was built with.
    surface_attraction: float = 0.6
    #: rescale everything about its centroid each step so the link's total
    #: length is preserved.  Without it the equilibrium size is whatever
    #: H and K happen to balance at, and the knot inflates while it rounds out.
    preserve_link_length: bool = True


def _link_paths(mesh: Mesh) -> list[list[int]]:
    return mesh.boundary_loops()


def _edge_list(mesh: Mesh) -> np.ndarray:
    return np.array(sorted(mesh.edge_map().keys()), dtype=int)


def _spring_forces(
    positions: np.ndarray, edges: np.ndarray, scale: float, p: RelaxParams
) -> np.ndarray:
    """Attraction along every edge."""
    delta = positions[edges[:, 1]] - positions[edges[:, 0]]
    length = np.linalg.norm(delta, axis=1)
    length[length == 0] = 1e-12
    magnitude = p.attraction * (length / scale) ** (1 + p.beta)
    pull = (magnitude / length)[:, None] * delta
    force = np.zeros_like(positions)
    np.add.at(force, edges[:, 0], pull)
    np.add.at(force, edges[:, 1], -pull)
    return force


def _repulsion(
    positions: np.ndarray, index: np.ndarray, exclude: set[frozenset[int]],
    scale: float, p: RelaxParams
) -> np.ndarray:
    """Inverse-power repulsion between the given vertices, skipping neighbours."""
    force = np.zeros_like(positions)
    pts = positions[index]
    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(dist, np.inf)
    for a, b in exclude:
        ia, ib = int(a), int(b)
        dist[ia, ib] = dist[ib, ia] = np.inf
    magnitude = p.repulsion * (dist / scale) ** -(2 + p.alpha)
    contribution = (magnitude / dist)[:, :, None] * delta
    np.add.at(force, index, np.nansum(contribution, axis=1))
    return force


def _collision_mask(
    candidates: np.ndarray, positions: np.ndarray, loop: list[int],
    clearance: float
) -> np.ndarray:
    """Which proposed moves would bring a link vertex too close to a
    non-neighbouring segment of the same component.

    Vectorised over all vertices of the loop at once: distances from every
    candidate position to every segment, with each vertex's own neighbourhood
    masked out.  Scharein's argument is that clamping the step size and
    refusing moves that violate this clearance together keep the link embedded.
    """
    n = len(loop)
    a = positions[loop]
    b = np.roll(a, -1, axis=0)
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom == 0] = 1e-12
    rel = candidates[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("kij,ij->ki", rel, ab) / denom, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    dist = np.linalg.norm(candidates[:, None, :] - closest, axis=2)
    idx = np.arange(n)
    for offset in (-2, -1, 0, 1):
        dist[idx, (idx + offset) % n] = np.inf
    return np.any(dist < clearance, axis=1)


def relax(
    mesh: Mesh,
    params: RelaxParams | None = None,
    iterations: int = 200,
    callback: Callable[[int, Mesh], None] | None = None,
) -> Mesh:
    """Run one relaxation cycle and return the relaxed mesh.

    There is no convergence test; the decaying time step is what makes a cycle
    settle.

    >>> from seifert.build import seifert_surface
    >>> m = relax(seifert_surface("AAA"), iterations=20)
    >>> m.info().genus
    1
    """
    params = params or RelaxParams()
    positions = mesh.vertices.copy()
    edges = _edge_list(mesh)

    scale = float(np.mean(np.linalg.norm(
        positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)))
    max_move = params.max_move * scale
    clearance = params.min_clearance * scale

    loops = _link_paths(mesh)
    link_index = np.array(sorted({v for loop in loops for v in loop}), dtype=int)
    position_in_link = {v: i for i, v in enumerate(link_index)}
    link_neighbours = {
        frozenset((position_in_link[a], position_in_link[b]))
        for loop in loops
        for a, b in zip(loop, loop[1:] + loop[:1])
    }
    interior = np.setdiff1d(np.arange(len(positions)), link_index)

    link_edges = np.array(
        [(a, b) for loop in loops for a, b in zip(loop, loop[1:] + loop[:1])],
        dtype=int,
    ) if loops else np.zeros((0, 2), dtype=int)

    target_link_length = float(np.linalg.norm(
        positions[link_edges[:, 1]] - positions[link_edges[:, 0]], axis=1).sum()
    ) if len(link_edges) else 0.0

    velocity = np.zeros_like(positions)
    dt = params.time_step

    for step in range(iterations):
        # The surface follows the link but does not influence it, so the two
        # populations get their forces from disjoint sources.
        force = np.zeros_like(positions)
        force[interior] = params.surface_attraction * _spring_forces(
            positions, edges, scale, params)[interior]

        if params.pin_boundary or len(link_index) < 3:
            force[link_index] = 0.0
        else:
            link_force = _spring_forces(positions, link_edges, scale, params)
            link_force += _repulsion(
                positions, link_index, link_neighbours, scale, params)
            force[link_index] = link_force[link_index]

        velocity = (1 - params.damping) * velocity + force * dt
        move = velocity * dt
        length = np.linalg.norm(move, axis=1, keepdims=True)
        direction = np.divide(move, np.where(length == 0, 1.0, length))
        move = np.minimum(length, max_move) * direction

        if params.pin_boundary:
            move[link_index] = 0.0
            velocity[link_index] = 0.0
        elif clearance > 0:
            for loop in loops:
                idx = np.asarray(loop)
                blocked = _collision_mask(
                    positions[idx] + move[idx], positions, loop, clearance)
                move[idx[blocked]] = 0.0
                velocity[idx[blocked]] = 0.0

        positions += move

        if params.preserve_link_length and not params.pin_boundary and len(link_edges):
            current = float(np.linalg.norm(
                positions[link_edges[:, 1]] - positions[link_edges[:, 0]],
                axis=1).sum())
            if current > 1e-12:
                centroid = positions.mean(axis=0)
                positions = centroid + (positions - centroid) * (
                    target_link_length / current)

        dt *= 1 - params.step_decay

        if callback is not None:
            callback(step, Mesh(positions.copy(), list(mesh.faces),
                                list(mesh.face_groups)))

    return Mesh(positions, list(mesh.faces), list(mesh.face_groups))
