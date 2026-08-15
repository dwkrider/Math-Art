# Polylinks: rings threaded through one another with polyhedral symmetry.
#
# Part of the Math Art weaving engine (`math_art/weaving/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# A polylink replaces each face or edge orbit of a polyhedron with a
# ring, positioned so the rings interlock without touching -- the
# orthoscheme construction Alan Holden described as 'regular polylinks'.
#
# References:
# - A. Holden, "Orderly Tangles: Cloverleafs, Gordian Knots, and
#   Regular Polylinks", Columbia University Press, 1983.

import math
import numpy as np
from math import cos, sin, pi

try:                                  # inside the math_art package
    from ..curve_frames import closed_tube as _shared_closed_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import closed_tube as _shared_closed_tube
try:                                  # inside the math_art package
    from ..polyhedra.seeds import icosa_faces as _icosa_faces
    from ..polyhedra.seeds import seed_poly
except ImportError:                   # flat import (test runner)
    from polyhedra.seeds import icosa_faces as _icosa_faces
    from polyhedra.seeds import seed_poly


PHI = (1 + 5 ** 0.5) / 2


def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def _closed_tube(pts, tube_r, sides, verts, faces, face_tag, fidx):
    """Append a swept closed tube along `pts` into the caller's buffers.

    The sweep itself is shared (`curve_frames.closed_tube`); this wrapper
    only rebases the indices and carries the per-face tag.

    NOTE: this module previously carried its own copy, which measured the
    closure holonomy as `sign * acos(dot)`.  That recovers the wrong
    branch, so the correction did not actually close the loop and the
    whole residual -- about 1.7 radians on a typical ring -- landed on the
    single seam quad as a twist discontinuity.  The shared sweep uses
    atan2 over the full range and spreads the correction evenly.
    """
    P = np.asarray(pts, float)
    if len(P) < 3:
        return
    base = len(verts)
    v, f = _shared_closed_tube(P, tube_r, sides)
    verts.extend(v)
    for quad in f:
        faces.append([base + i for i in quad])
        face_tag.append(fidx)


def build_polylinks(kind='TETRA', size=1.35, rotation=25.0, offset=-0.2,
                    width=0.14, thickness=0.10, antipodal=False, scale=1.0,
                    link_shape='POLYGON', amplitude=0.35, wave_factor=1,
                    knot_p=1, knot_q_factor=1, tube_sides=8,
                    segments=128):
    """One link per (selected) face of the solid: a flat polygon
    frame (classic), a radius-modulated wavy circle, or a torus
    knot about the face axis (the latter two after Shengyi Wang's
    polylink add-on)."""
    V, F = seed_poly(kind)
    if antipodal:
        keep = []
        used = []
        for f in F:
            c = _unit([sum(V[i][k] for i in f) / len(f) for k in range(3)])
            if any(abs(c[0] + u[0]) + abs(c[1] + u[1]) + abs(c[2] + u[2])
                   < 1e-6 for u in used):
                continue
            used.append(c)
            keep.append(f)
        F = keep
    verts = []
    faces = []
    face_frame = []      # frame index per emitted mesh face
    frame_dirs = []      # frame normal per frame (for pair grouping)
    rot = math.radians(rotation)
    for fidx, f in enumerate(F):
        m = len(f)
        c = [sum(V[i][k] for i in f) / m for k in range(3)]
        n = _unit(c)                      # Platonic: normal is radial
        frame_dirs.append(n)
        cl = math.sqrt(sum(x * x for x in c))
        cen = [n[k] * (cl + offset) for k in range(3)]
        if link_shape in ('WAVE', 'KNOT'):
            d0 = [V[f[0]][k] - c[k] for k in range(3)]
            rad = math.sqrt(sum(x * x for x in d0)) * size
            rot0 = math.radians(rotation)
            xr = _unit(d0)
            # rotate xr about n by rotation (Rodrigues)
            dd = sum(xr[k] * n[k] for k in range(3))
            cr = (n[1] * xr[2] - n[2] * xr[1],
                  n[2] * xr[0] - n[0] * xr[2],
                  n[0] * xr[1] - n[1] * xr[0])
            xN = [xr[k] * cos(rot0) + cr[k] * sin(rot0)
                  + n[k] * dd * (1 - cos(rot0)) for k in range(3)]
            yN = (n[1] * xN[2] - n[2] * xN[1],
                  n[2] * xN[0] - n[0] * xN[2],
                  n[0] * xN[1] - n[1] * xN[0])
            pts = []
            if link_shape == 'WAVE':
                frq = max(1, wave_factor) * m
                for i in range(segments):
                    t = 2 * pi * i / segments
                    rr = rad + amplitude * cos(frq * t)
                    pts.append(tuple(
                        (cen[k] + rr * (cos(t) * xN[k]
                                        + sin(t) * yN[k])) * scale
                        for k in range(3)))
            else:
                q = max(1, knot_q_factor) * m
                p = max(1, knot_p)
                r2 = amplitude
                for i in range(segments):
                    t = 2 * pi * i / segments
                    pts.append(tuple(
                        (cen[k]
                         + (rad + r2 * cos(q * t))
                         * (cos(p * t) * xN[k] + sin(p * t) * yN[k])
                         + r2 * sin(q * t) * n[k]) * scale
                        for k in range(3)))
            _closed_tube(pts, thickness / 2 * scale, tube_sides,
                         verts, faces, face_frame, fidx)
            continue
        ring = []
        for i in f:
            d = [V[i][k] - c[k] for k in range(3)]
            # rotate d about n by rot (Rodrigues), then scale
            dd = sum(d[k] * n[k] for k in range(3))
            cr = (n[1] * d[2] - n[2] * d[1], n[2] * d[0] - n[0] * d[2],
                  n[0] * d[1] - n[1] * d[0])
            r = [(d[k] * cos(rot) + cr[k] * sin(rot)
                  + n[k] * dd * (1 - cos(rot))) * size for k in range(3)]
            ring.append(r)
        base = len(verts)
        inner = 1.0 - width
        for layer in (thickness / 2, -thickness / 2):
            for r in ring:
                verts.append(tuple((cen[k] + r[k] + n[k] * layer) * scale
                                   for k in range(3)))
            for r in ring:
                verts.append(tuple((cen[k] + r[k] * inner + n[k] * layer)
                                   * scale for k in range(3)))
        TO, TI, BO, BI = base, base + m, base + 2 * m, base + 3 * m
        for i in range(m):
            j = (i + 1) % m
            faces.append([TO + i, TO + j, TI + j, TI + i])   # top
            faces.append([BI + i, BI + j, BO + j, BO + i])   # bottom
            faces.append([TO + j, TO + i, BO + i, BO + j])   # outer wall
            faces.append([TI + i, TI + j, BI + j, BI + i])   # inner wall
            face_frame.extend([fidx] * 4)
    return verts, faces, len(F), face_frame, frame_dirs
