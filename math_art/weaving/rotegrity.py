# Rotegrities: rotational tensegrities of tilted struts.
#
# Part of the Math Art weaving engine (`math_art/weaving/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Each strut is rotated in its own plane so that its ends bear on the
# neighbours' ends, giving a self-supporting ring structure with no
# strut touching another at its centre.
#
# References:
# - D. G. Emmerich, "Structures Tendues et Autotendantes", 1988.
# - C. Kitrick, "A unified approach to class I, II & III geodesic
#   domes", International Journal of Space Structures 5, 1990.

import math
from math import sin, cos, pi

try:                                  # inside the math_art package
    from ..polyhedra.seeds import icosa_faces as _icosa_faces
    from ..polyhedra.seeds import seed_poly as _shared_seed
except ImportError:                   # flat import (test runner)
    from polyhedra.seeds import icosa_faces as _icosa_faces
    from polyhedra.seeds import seed_poly as _shared_seed


PHI = (1 + 5 ** 0.5) / 2


def seed_poly(kind):
    """Platonic seed normalised to unit circumradius.

    This module and two others built their seeds this way, while
    three others used the raw coordinates.  The two sets are
    EXACTLY related by that scale (verified vertex for vertex on
    all five solids, with identical face lists), so the shared
    table carries both behind its `unit` flag and this wrapper
    keeps the call sites here reading as before.
    """
    return _shared_seed(kind, unit=True)


def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def geodesic(V, F, freq):
    """Class-I geodesic subdivision of a triangular polyhedron,
    projected to the unit sphere."""
    if freq <= 1:
        return V, F
    verts = list(V)
    key = {}
    faces = []

    def vid(p):
        k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if k not in key:
            key[k] = len(verts)
            verts.append(p)
        return key[k]

    for f in F:
        if len(f) != 3:
            raise ValueError("geodesic subdivision needs triangles")
        A, B, C = (V[i] for i in f)
        grid = {}
        for i in range(freq + 1):
            for j in range(freq + 1 - i):
                k = freq - i - j
                p = _unit(tuple((i * A[c] + j * B[c] + k * C[c]) / freq
                                for c in range(3)))
                grid[(i, j)] = vid(p)
        for i in range(freq):
            for j in range(freq - i):
                faces.append([grid[(i, j)], grid[(i + 1, j)],
                              grid[(i, j + 1)]])
                if j < freq - i - 1:
                    faces.append([grid[(i + 1, j)], grid[(i + 1, j + 1)],
                                  grid[(i, j + 1)]])
    return verts, faces


def _rodrigues(p, axis, ang):
    c, s = cos(ang), sin(ang)
    d = sum(axis[k] * p[k] for k in range(3))
    cr = (axis[1] * p[2] - axis[2] * p[1],
          axis[2] * p[0] - axis[0] * p[2],
          axis[0] * p[1] - axis[1] * p[0])
    return tuple(p[k] * c + cr[k] * s + axis[k] * d * (1 - c)
                 for k in range(3))


def _slerp(a, b, t):
    d = max(-1.0, min(1.0, sum(a[k] * b[k] for k in range(3))))
    om = math.acos(d)
    if om < 1e-9:
        return a
    sa = sin((1 - t) * om) / sin(om)
    sb = sin(t * om) / sin(om)
    return _unit(tuple(a[k] * sa + b[k] * sb for k in range(3)))


def build_rotegrity(kind='ICOSA', freq=2, twist=18.0, extension=0.35,
                    width=0.06, thickness=0.025, segments=12, scale=1.0):
    """One closed strap solid per edge of the (geodesic) seed."""
    V, F = seed_poly(kind)
    if kind in ('TETRA', 'OCTA', 'ICOSA'):
        V, F = geodesic(V, F, freq)
    edges = set()
    for f in F:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            edges.add((min(a, b), max(a, b)))
    verts = []
    faces = []
    face_strap = []           # strap index per face
    strap_len = []            # arc length per strap (for classing)
    tw = math.radians(twist)
    for si, (a, b) in enumerate(sorted(edges)):
        A, B = _unit(V[a]), _unit(V[b])
        strap_len.append(math.acos(max(-1.0, min(1.0,
            sum(A[k] * B[k] for k in range(3))))))
        mid = _unit(tuple(A[k] + B[k] for k in range(3)))
        A2 = _rodrigues(A, mid, tw)
        B2 = _rodrigues(B, mid, tw)
        pole = _unit((A2[1] * B2[2] - A2[2] * B2[1],
                      A2[2] * B2[0] - A2[0] * B2[2],
                      A2[0] * B2[1] - A2[1] * B2[0]))
        base = len(verts)
        n_arc = segments
        r_in = 1.0 - thickness / 2
        r_out = 1.0 + thickness / 2
        ring = []
        for i in range(n_arc + 1):
            t = -extension + (1 + 2 * extension) * i / n_arc
            p = _slerp(A2, B2, t)
            qp = _unit(tuple(p[k] + pole[k] * (width / 2)
                             for k in range(3)))
            qm = _unit(tuple(p[k] - pole[k] * (width / 2)
                             for k in range(3)))
            row = [tuple(c * r_out * scale for c in qp),
                   tuple(c * r_out * scale for c in qm),
                   tuple(c * r_in * scale for c in qm),
                   tuple(c * r_in * scale for c in qp)]
            ring.append([base + 4 * i + j for j in range(4)])
            verts.extend(row)
        for i in range(n_arc):
            r0, r1 = ring[i], ring[i + 1]
            for j in range(4):
                faces.append([r0[j], r0[(j + 1) % 4],
                              r1[(j + 1) % 4], r1[j]])
            face_strap.extend([si] * 4)
        faces.append([ring[0][3], ring[0][2], ring[0][1], ring[0][0]])
        faces.append(list(ring[-1]))
        face_strap.extend([si] * 2)
    return verts, faces, len(edges), face_strap, strap_len
