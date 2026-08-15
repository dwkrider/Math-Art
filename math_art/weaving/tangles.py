# Orderly tangles: symmetric arrangements of interwoven polygons.
#
# Part of the Math Art weaving engine (`math_art/weaving/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# References:
# - A. Holden, "Orderly Tangles", Columbia University Press, 1983.

import math
from math import cos, sin, pi

try:
    from ..conway_operators import _hull_faces
except ImportError:
    try:
        from conway_operators import _hull_faces
    except ImportError:
        _hull_faces = None


PHI = (1 + 5 ** 0.5) / 2


def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def _axis_rot(axis, ang):
    x, y, z = _unit(axis)
    c, s = cos(ang), sin(ang)
    C = 1 - c
    return ((c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C))


def _rot_pts(R, pts):
    return [tuple(sum(R[i][k] * p[k] for k in range(3)) for i in range(3))
            for p in pts]


def _tetra(mirror=False):
    V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    if mirror:
        V = [(-x, y, z) for (x, y, z) in V]
        F = [(0, 2, 1), (0, 3, 2), (0, 1, 3), (1, 2, 3)]
    else:
        F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    return V, F


def _cube():
    V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
         (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return V, F


def _octa():
    V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
         (0, 0, 1), (0, 0, -1)]
    F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
         (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return V, F


_FIVE_AXIS = (0.0, 1.0, PHI)      # a 5-fold axis of the icosahedral group


def compound(kind):
    """List of (verts, faces, own_axis) per component. own_axis is a
    symmetry axis of that component, used for per-component rotation
    (the parameter that generates Lang-style polypolyhedra variants)."""
    out = []
    diag = (1, 1, 1)
    if kind == 'T2':                      # stella octangula
        V, F = _tetra(False)
        out.append((V, F, diag))
        V, F = _tetra(True)
        out.append((V, F, (-1, 1, 1)))
    elif kind in ('T5', 'T10'):
        mirrors = (False, True) if kind == 'T10' else (False,)
        for mir in mirrors:
            V0, F = _tetra(mir)
            ax0 = (-1, 1, 1) if mir else diag
            for k in range(5):
                R = _axis_rot(_FIVE_AXIS, 2 * pi * k / 5)
                out.append((_rot_pts(R, V0), F, _rot_pts(R, [ax0])[0]))
    elif kind in ('C5', 'O5'):
        V0, F = _cube() if kind == 'C5' else _octa()
        for k in range(5):
            R = _axis_rot(_FIVE_AXIS, 2 * pi * k / 5)
            out.append((_rot_pts(R, V0), F, _rot_pts(R, [diag])[0]))
    elif kind == 'C3':                    # Escher's compound
        V, F = _cube()
        out.append((V, F, (0, 0, 1)))
        for axis in ((1, 0, 0), (0, 1, 0)):
            R = _axis_rot(axis, pi / 4)
            out.append((_rot_pts(R, V), F, axis))
    else:
        raise ValueError(kind)
    return out


def _emit_face_rings(verts, faces, face_comp, ci, V, F, width,
                     thickness, scale):
    """Leonardo da Vinci style: the panel shell between the
    polyhedron scaled out and in by half the thickness, with a hole
    inset in every face. Adjacent panels share the scaled polyhedron
    vertices, so the joints along edges and at vertices are exact
    (watertight mitres), as in the models Leonardo drew for Pacioli's
    De divina proportione."""
    k_out = 1.0 + thickness / 2
    k_in = max(0.05, 1.0 - thickness / 2)
    inner = 1.0 - width
    base_o = len(verts)
    verts.extend(tuple(x * k_out * scale for x in v) for v in V)
    base_i = len(verts)
    verts.extend(tuple(x * k_in * scale for x in v) for v in V)
    for f in F:
        m = len(f)
        c = [sum(V[i][k] for i in f) / m for k in range(3)]
        hole = [[c[k] + (V[i][k] - c[k]) * inner for k in range(3)]
                for i in f]
        HO = len(verts)
        verts.extend(tuple(p[k] * k_out * scale for k in range(3))
                     for p in hole)
        HI = len(verts)
        verts.extend(tuple(p[k] * k_in * scale for k in range(3))
                     for p in hole)
        for i in range(m):
            j = (i + 1) % m
            a, b = f[i], f[j]
            faces.append([base_o + a, base_o + b, HO + j, HO + i])
            faces.append([HI + i, HI + j, base_i + b, base_i + a])
            faces.append([HO + j, HO + i, HI + i, HI + j])
            face_comp.extend([ci] * 3)


def _emit_edge_struts(verts, faces, face_comp, ci, V, F, thickness,
                      scale, cap_size=1.0):
    """Square struts trimmed back from the vertices and closed with
    flat caps; each vertex gets a faceted knuckle (the convex hull of
    the surrounding strut caps) so the joints are clean."""
    h = thickness / 2
    edges = set()
    for f in F:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            edges.add((min(a, b), max(a, b)))
    vertex_rings = {}          # vertex index -> knuckle corner points
    for a, b in edges:
        A, B = V[a], V[b]
        t = _unit(tuple(B[k] - A[k] for k in range(3)))
        elen = math.sqrt(sum((B[k] - A[k]) ** 2 for k in range(3)))
        s = min(thickness * cap_size, 0.35 * elen)
        ref = (0, 0, 1) if abs(t[2]) < 0.9 else (1, 0, 0)
        u = _unit((t[1] * ref[2] - t[2] * ref[1],
                   t[2] * ref[0] - t[0] * ref[2],
                   t[0] * ref[1] - t[1] * ref[0]))
        w = (t[1] * u[2] - t[2] * u[1], t[2] * u[0] - t[0] * u[2],
             t[0] * u[1] - t[1] * u[0])
        A2 = tuple(A[k] + t[k] * s for k in range(3))
        B2 = tuple(B[k] - t[k] * s for k in range(3))
        base = len(verts)
        for P, vi in ((A2, a), (B2, b)):
            ring = []
            for (su, sw) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                p = tuple((P[k] + (u[k] * su + w[k] * sw) * h) * scale
                          for k in range(3))
                verts.append(p)
                ring.append(p)
            vertex_rings.setdefault(vi, []).extend(ring)
        for i2 in range(4):
            j2 = (i2 + 1) % 4
            faces.append([base + i2, base + j2, base + 4 + j2,
                          base + 4 + i2])
        faces.append([base + 3, base + 2, base + 1, base + 0])
        faces.append([base + 4, base + 5, base + 6, base + 7])
        face_comp.extend([ci] * 6)
    if _hull_faces is None:
        return
    for vi, pts in vertex_rings.items():
        c = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
        local = [tuple(p[k] - c[k] for k in range(3)) for p in pts]
        base = len(verts)
        verts.extend(tuple(p[k] + c[k] for k in range(3))
                     for p in local)
        for hf in _hull_faces(local):
            faces.append([base + i for i in hf])
            face_comp.append(ci)


def _emit_ball_stick(verts, faces, face_comp, ci, V, F, strut_radius,
                     node_radius, scale):
    """Round ball-and-stick frame for one component: a cylinder per
    edge and a sphere per vertex, via the shared ball_and_stick module
    (the same struts and nodes every polyhedron generator uses)."""
    try:
        from .styles import ball_and_stick
    except ImportError:
        from styles import ball_and_stick
    Vs = [tuple(x * scale for x in v) for v in V]
    edges = ball_and_stick.edges_from_faces(F)
    bv, bf = ball_and_stick.build_mesh(Vs, edges, strut_radius,
                                       node_radius)
    base = len(verts)
    verts.extend(bv)
    for f in bf:
        faces.append([base + i for i in f])
    face_comp.extend([ci] * len(bf))


def build_tangle(kind='T5', style='FACES', width=0.22, thickness=0.10,
                 size=1.0, comp_rot=0.0, spin=0.0, scale=1.0,
                 cap_size=1.0, strut_radius=0.02, node_radius=0.035):
    """Frames for every component of the compound: hollow faces, edge
    struts, or a round ball-and-stick frame. comp_rot rotates each
    component about its own symmetry axis (Lang-style variants).
    Returns (verts, faces, n_components, face_comp)."""
    comps = compound(kind)
    if abs(comp_rot) > 1e-9:
        cr = math.radians(comp_rot)
        comps = [(_rot_pts(_axis_rot(ax, cr), V), F, ax)
                 for (V, F, ax) in comps]
    if abs(spin) > 1e-9:
        R = _axis_rot(_FIVE_AXIS if kind in ('T5', 'T10', 'C5', 'O5')
                      else (0, 0, 1), math.radians(spin))
        comps = [(_rot_pts(R, V), F, ax) for (V, F, ax) in comps]
    verts = []
    faces = []
    face_comp = []
    for ci, (V, F, _ax) in enumerate(comps):
        V = [tuple(x * size for x in v) for v in V]
        if style == 'EDGES':
            _emit_edge_struts(verts, faces, face_comp, ci, V, F,
                              thickness, scale, cap_size)
        elif style == 'BALLSTICK':
            _emit_ball_stick(verts, faces, face_comp, ci, V, F,
                             strut_radius, node_radius, scale)
        else:
            _emit_face_rings(verts, faces, face_comp, ci, V, F, width,
                             thickness, scale)
    return verts, faces, len(comps), face_comp
