"""Kauffman states, and the non-orientable spanning surfaces they give.

Seifert's algorithm is one choice among many.  At each crossing of a diagram
there are two ways to resolve it, and *any* choice gives a set of state circles
which, filled with disks and reconnected by a half-twisted band at each
crossing, spans the same link.  A diagram with c crossings therefore carries
2**c spanning surfaces, of which the Seifert surface is a single one.

    chi(S)                = #state circles - c
    S is orientable       <=>  the state graph is bipartite
    boundary of S         =  the link, for every state

The second line is the interesting one.  Every band here is half-twisted, so the
twist cochain is tau = 1 identically; whether that cochain is a *coboundary* --
whether the twist can be gauged away by flipping disks over -- is exactly whether
the state graph is two-colourable.  The Seifert state always is.  Most other
states are not, and those are the one-sided spanning surfaces.

Worked example, the trefoil as the braid AAA.  The all-Seifert state gives two
state circles joined by three bands (a theta graph: bipartite, orientable,
genus 1) -- the Seifert surface.  The all-turnback state gives three circles
joined in a triangle (an odd cycle, so not bipartite): chi = 0, one boundary
component, non-orientable genus 1.  That is the Mobius band with three
half-twists, whose boundary is the trefoil.

Reference: E. Kalfagianni, "State surfaces of links", arXiv:1804.05281, Lemma
2.2 for the bipartite criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Sequence

import numpy as np

from .braid import Braid

__all__ = ["State", "StateSurface", "state_data", "all_states",
           "seifert_state", "turnback_state", "minimal_crosscap_state",
           "boundary_count"]

#: One bit per crossing.  ``False`` keeps the oriented (Seifert) resolution,
#: ``True`` takes the turnback resolution that joins the two strands.
State = tuple[bool, ...]


def boundary_count(
    feet: Sequence[int],
    bands: Sequence[tuple[int, int, int, int]],
    reversed_disks: Sequence[bool] | None = None,
    twisted: bool | Sequence[bool] = True,
) -> int:
    """Boundary components of a ribbon graph whose every band is half-twisted.

    Darts are ``(disk, foot)``; a dart is followed across its band, flipping the
    side because the band is twisted, and then stepped to the next foot around
    the far disk.  The orbits of that permutation pair up, one pair per boundary
    circle.  ``reversed_disks`` flips the cyclic order at the given disks, which
    is the choice of which way round each state circle is traversed.
    """
    reversed_disks = list(reversed_disks or [False] * len(feet))
    if isinstance(twisted, bool):
        twisted = [twisted] * len(bands)
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    flip: dict[tuple[int, int], bool] = {}
    for (a, fa, b, fb), t in zip(bands, twisted):
        partner[(a, fa)] = (b, fb)
        partner[(b, fb)] = (a, fa)
        flip[(a, fa)] = flip[(b, fb)] = bool(t)

    def step(dart, direction):
        disk, foot = partner[dart]
        turn = direction * (-1 if reversed_disks[disk] else 1)
        return (disk, (foot + turn) % feet[disk]), direction

    seen, orbits = set(), 0
    for dart in partner:
        for side in (1, -1):
            if (dart, side) in seen:
                continue
            orbits += 1
            state = (dart, side)
            while state not in seen:
                seen.add(state)
                side = -state[1] if flip[state[0]] else state[1]
                moved, side_now = step(state[0], side)
                state = (moved, side_now)
    return orbits // 2


@dataclass(frozen=True)
class StateSurface:
    """Combinatorics of one state's spanning surface, before any geometry."""

    braid: Braid
    state: State
    n_circles: int
    #: one entry per crossing: (circle_a, foot_a, circle_b, foot_b, sign)
    bands: tuple[tuple[int, int, int, int, int], ...]
    #: number of band feet on each circle, in the traced cyclic order
    feet: tuple[int, ...]
    #: mean (row, level) of each circle, used to order the disks in space
    anchors: tuple[tuple[float, float], ...]

    @property
    def euler_characteristic(self) -> int:
        return self.n_circles - len(self.bands)

    @property
    def n_boundaries(self) -> int:
        """Every state surface spans the same link."""
        return self.braid.n_components

    @property
    def orientable(self) -> bool:
        """True iff the state graph is bipartite (Kalfagianni, Lemma 2.2)."""
        colour: dict[int, int] = {}
        adjacency: dict[int, list[int]] = {i: [] for i in range(self.n_circles)}
        for a, _, b, _, _ in self.bands:
            adjacency[a].append(b)
            adjacency[b].append(a)
        for root in range(self.n_circles):
            if root in colour:
                continue
            colour[root] = 0
            stack = [root]
            while stack:
                v = stack.pop()
                for w in adjacency[v]:
                    if w not in colour:
                        colour[w] = 1 - colour[v]
                        stack.append(w)
                    elif colour[w] == colour[v]:
                        return False
        return True

    @property
    def euler_genus(self) -> int:
        return 2 - self.euler_characteristic - self.n_boundaries

    @property
    def genus(self) -> int | None:
        """Orientable genus, or ``None`` when the surface is one-sided."""
        if not self.orientable:
            return None
        g, r = divmod(self.euler_genus, 2)
        return g if r == 0 else None

    @property
    def crosscap_number(self) -> int | None:
        """Non-orientable genus, or ``None`` when the surface is two-sided."""
        return None if self.orientable else self.euler_genus

    @property
    def twisted(self) -> bool:
        """Whether the realisation needs tau = 1 on every band.

        A one-sided state surface does: with tau = 0 the surface would be
        orientable whatever the rotation.  A two-sided one does not, and tau = 0
        is the nicer realisation because it needs only a *half* geometric twist
        per band rather than a full one -- which is exactly the classical
        picture of a Seifert band.
        """
        return not self.orientable

    def rotation_reversals(self) -> tuple[bool, ...]:
        """Which state circles to traverse the other way round.

        Tracing a cycle gives its band order only up to direction, and the
        directions have to agree with a single planar orientation or the
        boundary comes out wrong.  Rather than reconstruct the embedding, fix
        the directions by the property that defines a state surface: its
        boundary is the link.  Reversing every circle at once is a global
        reflection, so circle 0 can be held fixed.
        """
        m = self.n_circles
        plain = [(a, fa, b, fb) for a, fa, b, fb, _ in self.bands]
        if m > 18:                                   # search would be silly
            return (False,) * m
        for bits in product((False, True), repeat=m - 1):
            flags = (False,) + bits
            if boundary_count(self.feet, plain, flags,
                              self.twisted) == self.n_boundaries:
                return flags
        raise ValueError(
            f"no traversal of {self.state} gives {self.n_boundaries} boundary "
            f"component(s); the state tracing is wrong"
        )

    def spatial_order(self) -> list[int]:
        """Order the disks along the stacking axis.

        Spectral (Fiedler) ordering of the state graph, which places circles
        joined by a band close together and so keeps the bands short.  For the
        Seifert state the graph is a path and this recovers the row order,
        i.e. the paper's stacked layout.
        """
        n = self.n_circles
        if n < 3:
            return list(range(n))
        laplacian = np.zeros((n, n))
        for a, _, b, _, _ in self.bands:
            if a == b:
                continue
            laplacian[a, b] -= 1
            laplacian[b, a] -= 1
            laplacian[a, a] += 1
            laplacian[b, b] += 1
        values, vectors = np.linalg.eigh(laplacian)
        fiedler = vectors[:, 1] if len(values) > 1 else np.zeros(n)
        # break ties with the diagram anchors so the result is deterministic
        keys = [(round(float(fiedler[i]), 9), self.anchors[i]) for i in range(n)]
        return sorted(range(n), key=lambda i: keys[i])


def _resolved_diagram(braid: Braid, state: State):
    """Edges of the resolved diagram, and which edges each band attaches to.

    Nodes are strand slots ``(level, row)``; every node has degree exactly two,
    so the edges form disjoint cycles -- the state circles.  Each band attaches
    to one edge on each of its two circles, and the order in which those
    attachment edges appear along a circle is the rotation system.
    """
    n, c = braid.strands, braid.n_bands
    node = lambda level, row: level * n + row          # noqa: E731
    edges: list[tuple[int, int]] = []
    attach: list[tuple[int, int]] = []                 # per crossing

    def add(u: int, v: int) -> int:
        edges.append((u, v))
        return len(edges) - 1

    for q, (crossing, turn) in enumerate(zip(braid.crossings, state), start=1):
        k = crossing.row
        horizontal = {}
        for row in range(n):
            if turn and row in (k, k + 1):
                continue
            horizontal[row] = add(node(q - 1, row), node(q, row))
        if turn:
            low = add(node(q - 1, k), node(q - 1, k + 1))      # cup
            high = add(node(q, k), node(q, k + 1))             # cap
            attach.append((low, high))
        else:
            attach.append((horizontal[k], horizontal[k + 1]))
    for row in range(n):
        add(node(c, row), node(0, row))                        # closure
    return edges, attach


def _trace_circles(n_nodes: int, edges: list[tuple[int, int]]):
    """Walk each cycle; return the circles as ordered edge sequences."""
    incident: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for e, (u, v) in enumerate(edges):
        incident[u].append(e)
        incident[v].append(e)

    seen_edge: set[int] = set()
    circles: list[list[int]] = []
    for start_edge in range(len(edges)):
        if start_edge in seen_edge:
            continue
        walk: list[int] = []
        edge, node = start_edge, edges[start_edge][0]
        while edge not in seen_edge:
            seen_edge.add(edge)
            walk.append(edge)
            u, v = edges[edge]
            node = v if node == u else u
            nxt = [e for e in incident[node] if e != edge]
            if not nxt:
                break
            edge = nxt[0]
        circles.append(walk)
    return circles


def state_data(braid: Braid | str, state: Sequence[bool]) -> StateSurface:
    """Resolve every crossing, trace the state circles, and read off the
    rotation system.

    The rotation matters: the Euler characteristic depends only on the counts,
    but the number of boundary components depends on the cyclic order in which
    bands meet each circle, and that order is what makes the boundary come out
    as the original link.
    """
    if isinstance(braid, str):
        braid = Braid(braid)
    state = tuple(bool(s) for s in state)
    n, c = braid.strands, braid.n_bands
    if len(state) != c:
        raise ValueError(f"state has {len(state)} bits, need {c}")

    edges, attach = _resolved_diagram(braid, state)
    circles = _trace_circles((c + 1) * n, edges)

    circle_of_edge = {e: i for i, walk in enumerate(circles) for e in walk}
    foot_of_edge: dict[int, int] = {}
    feet = [0] * len(circles)
    for i, walk in enumerate(circles):
        attaching = {e for pair in attach for e in pair}
        for e in walk:
            if e in attaching:
                foot_of_edge[e] = feet[i]
                feet[i] += 1

    bands = []
    for q, (crossing, (e_a, e_b)) in enumerate(zip(braid.crossings, attach)):
        bands.append((
            circle_of_edge[e_a], foot_of_edge[e_a],
            circle_of_edge[e_b], foot_of_edge[e_b],
            crossing.sign,
        ))

    node_of = lambda idx: divmod(idx, n)               # noqa: E731
    anchors = []
    for walk in circles:
        rows, levels = [], []
        for e in walk:
            for endpoint in edges[e]:
                level, row = node_of(endpoint)
                rows.append(row)
                levels.append(level)
        anchors.append((float(np.mean(rows)), float(np.mean(levels))))

    return StateSurface(braid, state, len(circles), tuple(bands),
                        tuple(feet), tuple(anchors))


def seifert_state(braid: Braid | str) -> State:
    """All crossings resolved the oriented way: Seifert's algorithm."""
    braid = Braid(braid) if isinstance(braid, str) else braid
    return (False,) * braid.n_bands


def turnback_state(braid: Braid | str) -> State:
    """The opposite extreme; for an alternating diagram this is the other
    checkerboard surface."""
    braid = Braid(braid) if isinstance(braid, str) else braid
    return (True,) * braid.n_bands


def all_states(braid: Braid | str) -> Iterator[StateSurface]:
    """Every one of the 2**c states.  Exponential -- guard the crossing count."""
    braid = Braid(braid) if isinstance(braid, str) else braid
    for bits in product((False, True), repeat=braid.n_bands):
        yield state_data(braid, bits)


def minimal_crosscap_state(braid: Braid | str) -> StateSurface:
    """The non-orientable state surface of smallest crosscap number.

    Its crosscap number is an upper bound for the link's, since the crosscap
    number is the minimum over *all* non-orientable spanning surfaces and state
    surfaces are only some of them.
    """
    braid = Braid(braid) if isinstance(braid, str) else braid
    best: StateSurface | None = None
    for s in all_states(braid):
        if s.orientable:
            continue
        if best is None or s.crosscap_number < best.crosscap_number:
            best = s
    if best is None:
        raise ValueError(f"no non-orientable state for {braid.word!r}")
    return best
