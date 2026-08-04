"""A minimal quad/polygon mesh with the topological queries this package needs.

Deliberately not a half-edge structure: a half-edge's ``twin`` relation encodes
a global orientation, so it cannot represent a non-orientable surface.  An
indexed face set plus an explicit orientation test costs nothing and stays
honest.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Mesh", "MeshInfo"]

Face = tuple[int, ...]


@dataclass(frozen=True)
class MeshInfo:
    """Topological summary of a mesh."""

    n_vertices: int
    n_edges: int
    n_faces: int
    euler_characteristic: int
    n_boundaries: int
    orientable: bool

    @property
    def euler_genus(self) -> int:
        return 2 - self.euler_characteristic - self.n_boundaries

    @property
    def genus(self) -> int | None:
        """Orientable genus, or ``None`` if the surface is one-sided."""
        if not self.orientable:
            return None
        g, r = divmod(self.euler_genus, 2)
        return g if r == 0 else None

    def __str__(self) -> str:
        side = "orientable" if self.orientable else "NON-orientable"
        extra = f"genus={self.genus}" if self.orientable else f"k={self.euler_genus}"
        return (
            f"V={self.n_vertices} E={self.n_edges} F={self.n_faces} "
            f"chi={self.euler_characteristic} b={self.n_boundaries} "
            f"{side} {extra}"
        )


@dataclass
class Mesh:
    """Vertices plus polygonal faces given as tuples of vertex indices."""

    vertices: np.ndarray
    faces: list[Face] = field(default_factory=list)
    #: optional per-face group label ("disk" / "band"), carried through refinement
    face_groups: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.faces = [tuple(int(i) for i in f) for f in self.faces]
        if self.face_groups and len(self.face_groups) != len(self.faces):
            raise ValueError("face_groups must be per-face")

    # -- construction -----------------------------------------------------
    @classmethod
    def empty(cls) -> "Mesh":
        return cls(np.zeros((0, 3)), [])

    def add_vertices(self, points: np.ndarray) -> np.ndarray:
        """Append points; return their new indices."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        start = len(self.vertices)
        self.vertices = np.vstack([self.vertices, points])
        return np.arange(start, start + len(points))

    def add_face(self, *indices: int, group: str | None = None) -> None:
        self.faces.append(tuple(int(i) for i in indices))
        if group is not None or self.face_groups:
            self.face_groups.append(group or "")

    # -- topology ---------------------------------------------------------
    def edge_map(self) -> dict[tuple[int, int], list[tuple[int, tuple[int, int]]]]:
        """Undirected edge -> list of (face index, directed edge as traversed)."""
        out: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
        for fi, f in enumerate(self.faces):
            n = len(f)
            for i in range(n):
                a, b = f[i], f[(i + 1) % n]
                out[(min(a, b), max(a, b))].append((fi, (a, b)))
        return out

    def boundary_edges(self) -> list[tuple[int, int]]:
        return [e for e, uses in self.edge_map().items() if len(uses) == 1]

    def boundary_loops(self) -> list[list[int]]:
        """Boundary components as ordered vertex-index cycles."""
        adjacency: dict[int, list[int]] = defaultdict(list)
        for a, b in self.boundary_edges():
            adjacency[a].append(b)
            adjacency[b].append(a)
        loops, visited = [], set()
        for start in adjacency:
            if start in visited:
                continue
            loop, previous, current = [start], None, start
            visited.add(start)
            while True:
                nxt = [v for v in adjacency[current] if v != previous]
                nxt = [v for v in nxt if v not in visited] or nxt
                if not nxt or nxt[0] == start:
                    break
                previous, current = current, nxt[0]
                visited.add(current)
                loop.append(current)
            if len(loop) > 2:
                loops.append(loop)
        return loops

    def orientation_signs(self) -> tuple[dict[int, int], int]:
        """Propagate an orientation over the dual graph.

        Returns the per-face flip sign and the number of conflict edges.  On a
        one-sided surface a coherent winding does not exist, so the signs are a
        best-effort partition whose only inconsistency is along the true seam.
        """
        dual: dict[int, list[tuple[int, bool]]] = defaultdict(list)
        for edge, uses in self.edge_map().items():
            if len(uses) != 2:
                continue
            (f1, d1), (f2, d2) = uses
            dual[f1].append((f2, d1 == d2))
            dual[f2].append((f1, d1 == d2))
        sign: dict[int, int] = {}
        conflicts = 0
        for root in range(len(self.faces)):
            if root in sign:
                continue
            sign[root] = 0
            stack = [root]
            while stack:
                a = stack.pop()
                for b, same in dual[a]:
                    want = sign[a] ^ int(same)
                    if b not in sign:
                        sign[b] = want
                        stack.append(b)
                    elif sign[b] != want:
                        conflicts += 1
        return sign, conflicts

    def is_orientable(self) -> bool:
        _, conflicts = self.orientation_signs()
        return conflicts == 0

    def oriented(self) -> "Mesh":
        """Rewind faces to a coherent winding (best effort if one-sided).

        The builder emits disks and bands independently, so their faces are
        wound at random relative to each other; with smooth shading that shows
        as dark creases along every disk/band seam, where the averaged vertex
        normal flips.  Rewinding by the propagated orientation removes them --
        fully for an orientable surface, everywhere but the unavoidable seam for
        a one-sided one.
        """
        sign, _ = self.orientation_signs()
        faces = [tuple(reversed(f)) if sign.get(i, 0) else f
                 for i, f in enumerate(self.faces)]
        return Mesh(self.vertices.copy(), faces, list(self.face_groups))

    def triangulated(self) -> "Mesh":
        """Fan-triangulate every face (for the cotangent Laplacian and area)."""
        tris, groups = [], []
        for fi, f in enumerate(self.faces):
            for i in range(1, len(f) - 1):
                tris.append((f[0], f[i], f[i + 1]))
                if self.face_groups:
                    groups.append(self.face_groups[fi])
        return Mesh(self.vertices.copy(), tris, groups)

    def area(self) -> float:
        tri = self.triangulated()
        f = np.array(tri.faces)
        if not len(f):
            return 0.0
        a, b, c = tri.vertices[f[:, 0]], tri.vertices[f[:, 1]], tri.vertices[f[:, 2]]
        return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)

    def info(self) -> MeshInfo:
        edges = self.edge_map()
        used = {i for f in self.faces for i in f}
        return MeshInfo(
            n_vertices=len(used),
            n_edges=len(edges),
            n_faces=len(self.faces),
            euler_characteristic=len(used) - len(edges) + len(self.faces),
            n_boundaries=len(self.boundary_loops()),
            orientable=self.is_orientable(),
        )
