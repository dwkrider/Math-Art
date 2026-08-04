"""Catmull-Clark refinement of the quad mesh.

Boundary rules follow the usual convention: a boundary edge point is the edge
midpoint, and a boundary vertex uses the cubic B-spline stencil (1, 6, 1)/8
along the boundary curve, so the rim converges to a smooth curve independent of
the interior.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .mesh import Mesh

__all__ = ["catmull_clark"]


def _topology(mesh: Mesh):
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, f in enumerate(mesh.faces):
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            edge_faces[(min(a, b), max(a, b))].append(fi)
    vertex_faces: dict[int, list[int]] = defaultdict(list)
    vertex_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for fi, f in enumerate(mesh.faces):
        for v in f:
            vertex_faces[v].append(fi)
    for e in edge_faces:
        vertex_edges[e[0]].append(e)
        vertex_edges[e[1]].append(e)
    return edge_faces, vertex_faces, vertex_edges


def catmull_clark(mesh: Mesh, levels: int = 1) -> Mesh:
    """Subdivide ``levels`` times; every face becomes ``len(face)`` quads.

    >>> from seifert.build import seifert_surface
    >>> before = seifert_surface("AAA").info()
    >>> after = catmull_clark(seifert_surface("AAA")).info()
    >>> before.euler_characteristic == after.euler_characteristic
    True
    """
    for _ in range(levels):
        mesh = _subdivide_once(mesh)
    return mesh


def _subdivide_once(mesh: Mesh) -> Mesh:
    edge_faces, vertex_faces, vertex_edges = _topology(mesh)
    V = mesh.vertices

    face_points = np.array([V[list(f)].mean(axis=0) for f in mesh.faces])

    edge_points: dict[tuple[int, int], np.ndarray] = {}
    for e, faces in edge_faces.items():
        midpoint = 0.5 * (V[e[0]] + V[e[1]])
        if len(faces) == 1:                       # boundary edge
            edge_points[e] = midpoint
        else:
            edge_points[e] = 0.5 * (midpoint + face_points[faces].mean(axis=0))

    boundary_neighbours: dict[int, list[int]] = defaultdict(list)
    for e, faces in edge_faces.items():
        if len(faces) == 1:
            boundary_neighbours[e[0]].append(e[1])
            boundary_neighbours[e[1]].append(e[0])

    new_vertex: dict[int, np.ndarray] = {}
    for v in range(len(V)):
        if v in boundary_neighbours:
            nb = boundary_neighbours[v]
            if len(nb) == 2:
                new_vertex[v] = (V[nb[0]] + 6 * V[v] + V[nb[1]]) / 8.0
            else:                                  # corner or non-manifold rim
                new_vertex[v] = V[v]
            continue
        faces = vertex_faces.get(v, [])
        edges = vertex_edges.get(v, [])
        n = len(faces)
        if n == 0:
            new_vertex[v] = V[v]
            continue
        F = face_points[faces].mean(axis=0)
        R = np.mean([0.5 * (V[a] + V[b]) for a, b in edges], axis=0)
        new_vertex[v] = (F + 2 * R + (n - 3) * V[v]) / n

    out = Mesh.empty()
    vertex_id = {v: int(out.add_vertices([p])[0]) for v, p in new_vertex.items()}
    face_id = {fi: int(out.add_vertices([p])[0]) for fi, p in enumerate(face_points)}
    edge_id = {e: int(out.add_vertices([p])[0]) for e, p in edge_points.items()}

    def key(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    for fi, f in enumerate(mesh.faces):
        n = len(f)
        group = mesh.face_groups[fi] if mesh.face_groups else None
        for i in range(n):
            prev_v, v, next_v = f[i - 1], f[i], f[(i + 1) % n]
            out.add_face(
                vertex_id[v],
                edge_id[key(v, next_v)],
                face_id[fi],
                edge_id[key(prev_v, v)],
                group=group,
            )
    return out
