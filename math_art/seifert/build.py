"""Build the Seifert surface of a braid closure as a quad mesh.

The construction follows section IV ("Structure") and V ("Geometry") of van Wijk
and Cohen: one disk per Seifert circle, stacked along z, joined by one twisted
band per letter of the braid word.

Two deliberate departures from the paper, both noted in the module that
implements them:

* The paper models each disk as a flattened *ellipsoid* -- a two-sided lens --
  and each band as a *tube* through holes cut in it, to colour the two sides
  differently.  This module builds the honest single-sheet surface directly.
* Bands are framed with a rotation-minimising frame rather than the Frenet
  frame (see :mod:`seifert.frames`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .braid import Braid
from .states import State, StateSurface, seifert_state, state_data
from .frames import bezier, rotation_minimising_frames, twist_to_close, unit
from .mesh import Mesh

__all__ = ["SurfaceParams", "spanning_surface", "seifert_surface",
           "state_surface"]


@dataclass(frozen=True)
class SurfaceParams:
    """Geometry knobs, in units of the disk radius.

    The paper publishes no numeric defaults; these ratios come from a reference
    surface with disks of diameter 60 spaced 83 apart and bands 20 wide:

        disk_spacing = 83 / 30 = 2.77 radii
        band_width   = 20 / 30 = 0.67 radii

    The stack is therefore much taller and the bands much narrower than the
    obvious guess.
    """

    disk_radius: float = 1.0
    #: centre-to-centre distance between neighbouring disks (83/30)
    disk_spacing: float = 2.77
    #: band width as a fraction of the disk radius (20/30)
    band_width: float = 0.85
    #: upper bound on the arc, as a fraction of one foot's angular sector
    arc_span: float = 0.75
    #: rim samples per band foot.  Must be even, so the O-grid core closes up;
    #: enough of them that ``band_width`` can be resolved.
    samples_per_sector: int = 12
    #: number of annular rings outside the disk's square core
    radial_rings: int = 2
    #: radius of the square core as a fraction of the disk radius
    core_fraction: float = 0.45
    #: samples along a band
    band_samples: int = 18
    #: Bezier control-point offset, as a multiple of the disk radius
    bulge: float = 0.85
    #: direction a band leaves its disk: 1 = radially outward, 0 = straight up
    #: the stack toward the disk it is going to.  The paper launches its tubes
    #: along the ellipsoid normal at the attachment point, which is radial only
    #: if the hole sits on the equator; a hole on the face gives a nearly axial
    #: launch and a much shorter, straighter band.
    band_launch: float = 1.0

    def __post_init__(self) -> None:
        if self.samples_per_sector % 2:
            raise ValueError("samples_per_sector must be even")

    def arc_samples(self, n_theta: int) -> int:
        """Rim samples spanned by one band foot, from the requested width."""
        chord = min(0.98, self.band_width / (2 * self.disk_radius))
        wanted = 2 * np.arcsin(chord) * n_theta / (2 * np.pi)
        ceiling = max(2, int(self.arc_span * self.samples_per_sector))
        return int(min(ceiling, max(2, round(wanted))))


def _coons(boundary: np.ndarray, m: int) -> np.ndarray:
    """Transfinite (Coons) interpolation of a square grid from its boundary.

    ``boundary`` holds the 4m boundary positions in counter-clockwise order
    starting at grid corner (0, 0); the result is the full (m+1) x (m+1) grid.
    """
    grid = np.zeros((m + 1, m + 1, 3))
    bottom = boundary[0 : m + 1]                                  # (i, 0)
    right = np.vstack([boundary[m : 2 * m + 1]])                  # (m, j)
    top = np.vstack([boundary[2 * m : 3 * m + 1]])[::-1]          # (i, m)
    left = np.vstack([boundary[3 * m :], boundary[:1]])[::-1]     # (0, j)
    for i in range(m + 1):
        u = i / m
        for j in range(m + 1):
            v = j / m
            grid[i, j] = (
                (1 - v) * bottom[i] + v * top[i]
                + (1 - u) * left[j] + u * right[j]
                - ((1 - u) * (1 - v) * bottom[0] + u * (1 - v) * bottom[m]
                   + (1 - u) * v * top[0] + u * v * top[m])
            )
    return grid


def _disk(
    mesh: Mesh,
    centre: float | np.ndarray,
    n_theta: int,
    params: SurfaceParams,
) -> np.ndarray:
    """Add one disk as an O-grid; return its rim vertex indices in angle order.

    A polar fan would put a vertex of valence ``n_theta`` at the centre.  Both
    Catmull-Clark and the relaxation behave badly around a vertex of valence 70,
    and it shows as a radial starburst crease that no amount of smoothing
    removes.  An O-grid -- a square patch in the middle, annular rings around it
    -- keeps every vertex at valence four except four corners at valence three.
    """
    if n_theta % 4:
        raise ValueError("n_theta must be a multiple of 4 for the O-grid")
    origin = (np.array([0.0, 0.0, float(centre)])
              if np.isscalar(centre) else np.asarray(centre, dtype=float))
    m = n_theta // 4
    angles = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    inner = params.disk_radius * params.core_fraction

    ring_positions = origin + np.column_stack(
        [inner * np.cos(angles), inner * np.sin(angles), np.zeros(n_theta)]
    )
    grid = _coons(ring_positions, m)

    # index the grid, reusing the boundary entries in their cyclic order
    ids = np.full((m + 1, m + 1), -1, dtype=int)
    walk = ([(i, 0) for i in range(m)] + [(m, j) for j in range(m)]
            + [(i, m) for i in range(m, 0, -1)] + [(0, j) for j in range(m, 0, -1)])
    core_ring = mesh.add_vertices(np.array([grid[i, j] for i, j in walk]))
    for (i, j), vid in zip(walk, core_ring):
        ids[i, j] = vid
    for i in range(1, m):
        for j in range(1, m):
            ids[i, j] = int(mesh.add_vertices([grid[i, j]])[0])
    for i in range(m):
        for j in range(m):
            mesh.add_face(ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1],
                          group="disk")

    rings = [core_ring]
    for k in range(1, params.radial_rings + 1):
        r = inner + (params.disk_radius - inner) * k / params.radial_rings
        rings.append(mesh.add_vertices(origin + np.column_stack(
            [r * np.cos(angles), r * np.sin(angles), np.zeros(n_theta)])))
    for a, b in zip(rings, rings[1:]):
        for i in range(n_theta):
            k = (i + 1) % n_theta
            mesh.add_face(a[i], b[i], b[k], a[k], group="disk")
    return rings[-1]


def _attachment_arc(rim: np.ndarray, angle: float, width: int) -> np.ndarray:
    """The ``width + 1`` rim vertices centred on ``angle``.

    Every disk is meshed at the same angular resolution, so an arc of a given
    sample count subtends the same chord wherever it lands -- which is what lets
    the two ends of a band weld one-to-one.
    """
    n = len(rim)
    start = int(round(angle * n / (2 * np.pi))) - width // 2
    return np.array([rim[(start + i) % n] for i in range(width + 1)])


def _band(
    mesh: Mesh,
    arc_a: np.ndarray,
    arc_b: np.ndarray,
    angle_a: float,
    angle_b: float,
    half_turns: int,
    span: float,
    params: SurfaceParams,
) -> None:
    """Sweep a twisted ribbon between two rim arcs, reusing their vertices.

    Note on the twist parity.  Both disks face +z and the band leaves one rim
    radially outward and arrives at the other radially inward, so the band makes
    a U-turn.  That U-turn already reverses the frame once.  An *odd* number of
    geometric half-turns therefore delivers the far end face-up (twist cochain
    tau = 0 across this band) and an *even* number delivers it face-down
    (tau = 1) -- the opposite of what the flat, coplanar picture would suggest.
    """
    out_a = np.array([np.cos(angle_a), np.sin(angle_a), 0.0])
    out_b = np.array([np.cos(angle_b), np.sin(angle_b), 0.0])
    tan_a = np.array([-np.sin(angle_a), np.cos(angle_a), 0.0])
    tan_b = np.array([-np.sin(angle_b), np.cos(angle_b), 0.0])

    start, end = mesh.vertices[arc_a], mesh.vertices[arc_b]
    p0, p3 = start.mean(axis=0), end.mean(axis=0)
    reach = params.bulge * params.disk_radius + 0.42 * span
    if np.linalg.norm(p3 - p0) < 1e-9:
        reach += params.disk_radius            # a band onto its own disk
    axis = np.array([0.0, 0.0, np.sign(p3[2] - p0[2]) or 1.0])
    lam = params.band_launch
    launch_a = unit(lam * out_a + (1 - lam) * axis)
    launch_b = unit(lam * out_b - (1 - lam) * axis)
    points, tangents = bezier(p0, p0 + reach * launch_a,
                              p3 + reach * launch_b, p3,
                              params.band_samples)
    normals = rotation_minimising_frames(points, tangents, tan_a)

    total_twist = twist_to_close(normals[-1], tan_b, tangents[-1], half_turns)
    far = arc_b[::-1] if half_turns % 2 else arc_b

    width_axis = tan_a
    normal_axis = np.cross(tangents[0], width_axis)
    local = start - p0
    profile = np.column_stack([local @ width_axis, local @ normal_axis])

    rows = [np.asarray(arc_a)]
    for step in range(1, params.band_samples):
        if step == params.band_samples - 1:
            rows.append(np.asarray(far))
            break
        t = step / (params.band_samples - 1)
        tangent = tangents[step]
        u = unit(normals[step] - (normals[step] @ tangent) * tangent)
        v = np.cross(tangent, u)
        axis_u = np.cos(total_twist * t) * u + np.sin(total_twist * t) * v
        axis_v = np.cross(tangent, axis_u)
        rows.append(mesh.add_vertices(
            points[step]
            + np.outer(profile[:, 0], axis_u)
            + np.outer(profile[:, 1], axis_v)))

    for a, b in zip(rows, rows[1:]):
        for i in range(len(a) - 1):
            mesh.add_face(a[i], a[i + 1], b[i + 1], b[i], group="band")


def spanning_surface(
    feet: Sequence[int],
    bands: Sequence[tuple[int, int, int, int, int]],
    params: SurfaceParams | None = None,
) -> Mesh:
    """Disks stacked along z, joined by twisted bands.

    ``feet[d]`` is the number of attachment arcs on disk ``d``, spaced evenly
    around its rim in the order the rotation system prescribes.  ``bands`` holds
    ``(disk_a, foot_a, disk_b, foot_b, half_turns)``; a band may join a disk to
    itself, and to any disk in the stack rather than only its neighbour, which
    is what state surfaces need and Seifert surfaces do not.
    """
    params = params or SurfaceParams()
    feet = list(feet)
    n_theta = max(4, max(feet, default=1) * params.samples_per_sector)
    width = params.arc_samples(n_theta)

    mesh = Mesh.empty()
    rims, angles = [], []
    for count in feet:
        rims.append(_disk(mesh, len(rims) * params.disk_spacing, n_theta, params))
        angles.append([2 * np.pi * (j + 0.5) / max(count, 1) for j in range(count)])

    for disk_a, foot_a, disk_b, foot_b, half_turns in bands:
        _band(
            mesh,
            _attachment_arc(rims[disk_a], angles[disk_a][foot_a], width),
            _attachment_arc(rims[disk_b], angles[disk_b][foot_b], width),
            angles[disk_a][foot_a],
            angles[disk_b][foot_b],
            half_turns,
            abs(disk_a - disk_b) * params.disk_spacing,
            params,
        )
    return mesh


#: Geometric half-twists per band.  The band's U-turn already reverses the frame
#: once, so an *odd* count delivers the far end face-up (tau = 0) and an even
#: count face-down (tau = 1).  A two-sided surface can use tau = 0, and one
#: half-twist per band is both the smaller deformation and the classical picture
#: of a Seifert band; a one-sided one needs tau = 1 and so a full twist.
BAND_HALF_TURNS = {False: 1, True: 2}


def seifert_surface(braid: Braid | str, params: SurfaceParams | None = None) -> Mesh:
    """Seifert surface of a braid closure, in the *stacked* disk/band style.

    >>> info = seifert_surface("AbAb").info()          # figure-eight
    >>> info.euler_characteristic, info.n_boundaries, info.genus
    (-1, 1, 1)
    """
    braid = Braid(braid) if isinstance(braid, str) else braid
    return state_surface(braid, seifert_state(braid), params)


def state_surface(
    braid: Braid | str,
    state: State | StateSurface | None = None,
    params: SurfaceParams | None = None,
) -> Mesh:
    """Spanning surface for one Kauffman state -- orientable or not.

    ``state`` defaults to the Seifert state.  The disks are ordered along the
    stacking axis by the state graph's Fiedler vector, which for the Seifert
    state recovers the paper's row-by-row stack.

    >>> from seifert.states import turnback_state
    >>> m = state_surface("AAA", turnback_state("AAA"))   # trefoil, all turnback
    >>> i = m.info()
    >>> i.euler_characteristic, i.n_boundaries, i.orientable, i.euler_genus
    (0, 1, False, 1)
    """
    braid = Braid(braid) if isinstance(braid, str) else braid
    if state is None:
        data = state_data(braid, seifert_state(braid))
    elif isinstance(state, StateSurface):
        data = state
    else:
        data = state_data(braid, state)

    order = data.spatial_order()
    height = {circle: level for level, circle in enumerate(order)}
    flip = data.rotation_reversals()
    feet = [0] * data.n_circles
    for circle, level in height.items():
        feet[level] = data.feet[circle]

    def place(circle: int, foot: int) -> int:
        n = data.feet[circle]
        return (n - 1 - foot) if flip[circle] else foot

    # The geometric twist parity that realises tau depends on how far the band
    # turns on its way over, which depends on band_launch.  Rather than reason
    # about it, build both parities and keep the one whose mesh matches the
    # combinatorial prediction.
    wanted = (data.euler_characteristic, data.n_boundaries, data.orientable)
    fallback = None
    for extra in (0, 1):
        turns = BAND_HALF_TURNS[data.twisted] + extra
        mesh = spanning_surface(
            feet,
            [(height[a], place(a, fa), height[b], place(b, fb), turns * sign)
             for a, fa, b, fb, sign in data.bands],
            params,
        )
        info = mesh.info()
        if (info.euler_characteristic, info.n_boundaries,
                info.orientable) == wanted:
            return mesh
        fallback = fallback or mesh
    raise ValueError(
        f"neither twist parity realises {wanted} for state {data.state}"
    )
