
# Fractal Polyhedron Generator for Blender
#
# Recursive clusters of Platonic solids, after the fractal-polyhedron
# construction on George W. Hart's pages: every generation places a
# scaled copy of the solid at each vertex, face or edge of the current
# copies. Vertex mode with scale 1/2 on a tetrahedron and parents
# removed is the classic Sierpinski tetrahedron; face mode grows
# spiky coral-like clusters.
#
# Crystal mode adds the Fathauer/Haüy fractal crystals: instead of
# copies at anchors, s-scaled copies are stacked face-to-face on every
# unoccupied face of the previous generation (all copies in one
# orientation, tetrahedra alternating). The convex hull then
# approaches a crystal form -- usually the dual of the seed -- whose
# facets carry Sierpinski-like patterns: 3 faces meeting at a hull
# vertex give Sierpinski triangles (cube s=1/2 -> octahedron), 4 give
# filled square arrays, and 5 at s=phi^-2 give Sierpinski pentagons
# (icosahedron -> dodecahedron). Cube s=1/3 gives a rhombic
# dodecahedron; s=1 with duplicates merged reproduces Haüy's classic
# layer-by-layer octahedron of equal cubes.
#
# References:
# - Recursive Platonic-solid clusters: after George W. Hart.
# - Sierpinski tetrahedron / triangle: Waclaw Sierpinski (1916),
#   self-similar fractal.
# - R.-J. Haüy, "Essai d'une théorie sur la structure des crystaux",
#   1784 -- crystal forms built by stacking congruent blocks
#   ("molécules intégrantes"); basis of the crystal presets.
# - R. W. Fathauer, H. Kaczmarski, N. Duchnowski, "A Fractal Crystal
#   Comprised of Cubes and Some Related Fractal Arrangements of other
#   Platonic Solids", Bridges 2008, pp. 289-296.
#   https://archive.bridgesmathart.org/2008/bridges2008-289.pdf
# - R. W. Fathauer, "Iterative Arrangements of Polyhedra --
#   Relationships to Classical Fractals and Haüy Constructions",
#   Bridges 2013, pp. 175-182.
#   https://archive.bridgesmathart.org/2013/bridges2013-175.pdf
# - C. Goodman-Strauss, "Dodecafoam and Substitution Tilings",
#   Computers & Graphics 23 (1999), pp. 917-924 -- the dodecahedron
#   arrangement here is a subset of dodecafoam.

bl_info = {
    "name": "Fractal Polyhedra",
    "author": "Math Art project (after George W. Hart)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Fractal Polyhedron",
    "description": "Recursive Platonic solid clusters and Haüy "
                   "fractal crystals",
    "category": "Add Mesh",
}

import math

PHI = (1 + 5 ** 0.5) / 2


def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def _icosa_faces(V):
    n = len(V)
    emin = None
    d2 = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = sum((V[i][k] - V[j][k]) ** 2 for k in range(3))
            d2[(i, j)] = d
            emin = d if emin is None else min(emin, d)
    adj = {i: set() for i in range(n)}
    for (i, j), d in d2.items():
        if d < emin * 1.2:
            adj[i].add(j)
            adj[j].add(i)
    fs = set()
    for i in range(n):
        for j in adj[i]:
            for k in adj[i] & adj[j]:
                fs.add(tuple(sorted((i, j, k))))
    faces = []
    for f in fs:
        a, b, c = (V[i] for i in f)
        nx = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
              (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
              (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        cen = [(a[k] + b[k] + c[k]) / 3 for k in range(3)]
        if sum(nx[k] * cen[k] for k in range(3)) < 0:
            faces.append([f[0], f[2], f[1]])
        else:
            faces.append(list(f))
    return faces


def seed_poly(kind):
    if kind == 'TETRA':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    elif kind == 'OCTA':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
        F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    elif kind == 'CUBE':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    elif kind == 'ICOSA':
        V = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = _icosa_faces(V)
    elif kind == 'DODECA':
        IV = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                IV += [(0, a, b), (a, b, 0), (b, 0, a)]
        IF = _icosa_faces(IV)
        V = [tuple(sum(IV[i][k] for i in f) / 3 for k in range(3))
             for f in IF]
        F = []
        for vi in range(len(IV)):
            adj = [fi for fi, f in enumerate(IF) if vi in f]
            nrm = _unit(IV[vi])
            u0 = V[adj[0]]
            u = [u0[k] - sum(u0[j] * nrm[j] for j in range(3)) * nrm[k]
                 for k in range(3)]
            w = (nrm[1] * u[2] - nrm[2] * u[1],
                 nrm[2] * u[0] - nrm[0] * u[2],
                 nrm[0] * u[1] - nrm[1] * u[0])
            def ang(fi):
                d = V[fi]
                return math.atan2(sum(d[k] * w[k] for k in range(3)),
                                  sum(d[k] * u[k] for k in range(3)))
            F.append(sorted(adj, key=ang))
    else:
        raise ValueError(kind)
    return [list(v) for v in V], [list(f) for f in F]


def _anchors(V, F, mode):
    if mode == 'VERTS':
        return [tuple(v) for v in V]
    if mode == 'EDGES':
        seen = set()
        out = []
        for f in F:
            m = len(f)
            for i in range(m):
                a, b = f[i], f[(i + 1) % m]
                k = (min(a, b), max(a, b))
                if k not in seen:
                    seen.add(k)
                    out.append(tuple((V[a][c] + V[b][c]) / 2
                                     for c in range(3)))
        return out
    # FACES
    return [tuple(sum(V[i][c] for i in f) / len(f) for c in range(3))
            for f in F]


MAX_COPIES = 30000


def _prism_seed(n):
    """Right n-gonal prism, circumradius 1, square side faces
    (height = polygon edge length), centred at the origin."""
    e = 2.0 * math.sin(math.pi / n)
    h = e / 2.0
    V = []
    for z in (-h, h):
        for i in range(n):
            a = 2.0 * math.pi * (i + 0.5) / n
            V.append((math.cos(a), math.sin(a), z))
    F = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        F.append([i, j, n + j, n + i])
    return [list(v) for v in V], F


def _crystal_seed(kind):
    if kind == 'HEX_PRISM':
        return _prism_seed(6)
    if kind == 'OCT_PRISM':
        return _prism_seed(8)
    return seed_poly(kind)


def _face_data(V, F):
    """(unit outward normal, apothem) of every face; the apothem is
    the origin-to-face-plane distance."""
    out = []
    for f in F:
        nx = ny = nz = 0.0
        m = len(f)
        for i in range(m):
            x1, y1, z1 = V[f[i]]
            x2, y2, z2 = V[f[(i + 1) % m]]
            nx += (y1 - y2) * (z1 + z2)
            ny += (z1 - z2) * (x1 + x2)
            nz += (x1 - x2) * (y1 + y2)
        nrm = _unit((nx, ny, nz))
        a = sum(nrm[k] * V[f[0]][k] for k in range(3))
        if a < 0:
            nrm = tuple(-c for c in nrm)
            a = -a
        out.append((nrm, a))
    return out


# Haüy/Fathauer crystal presets: (seed solid, scaling factor s).
# Hulls follow Fathauer (Bridges 2008, 2013): the number of faces
# meeting at a hull vertex decides the facet pattern -- 3 gives
# Sierpinski triangles, 4 filled arrays, 5 (at s = phi^-2)
# Sierpinski pentagons.
_CRYSTAL_PRESETS = {
    'CUBE_OCT':     ('CUBE', 0.5),         # octahedron hull
    'CUBE_RD':      ('CUBE', 1.0 / 3.0),   # rhombic-dodecahedron hull
    'HAUY_OCT':     ('CUBE', 1.0),         # Haüy 1784, equal cubes
    'DODECAFOAM':   ('DODECA', 2.0 - PHI),  # phi^-2; dodecafoam subset
    'DODECA_ICOSA': ('DODECA', 0.5),       # icosahedron hull
    'ICOSA_SP':     ('ICOSA', 2.0 - PHI),  # dodecahedron hull
    'HEX_DIPYR':    ('HEX_PRISM', 0.5),    # hexagonal dipyramid hull
    'OCT_DIPYR':    ('OCT_PRISM', 0.5),    # octagonal dipyramid hull
}


def _build_crystal(crystal, kind, child_scale, generations,
                   keep_parents):
    """Fathauer/Haüy face-growth engine: each generation stacks
    s-scaled copies face-to-face on every unoccupied face of the
    previous generation. All copies share one orientation (tetrahedra
    alternate between the two mirror orientations, after
    Gosper/Moravec). Coincident duplicates are merged, so s = 1
    reproduces Haüy's layer construction of equal blocks."""
    if crystal == 'CUSTOM':
        ck, s = kind, child_scale
    else:
        ck, s = _CRYSTAL_PRESETS[crystal]
    V, F = _crystal_seed(ck)
    fdata = _face_data(V, F)
    alternate = (ck == 'TETRA')

    def key(c, sz):
        return (round(c[0], 6), round(c[1], 6), round(c[2], 6),
                round(sz, 9))

    frontier = [((0.0, 0.0, 0.0), 1.0, 1, 0, None)]
    all_copies = list(frontier)
    seen = {key((0.0, 0.0, 0.0), 1.0)}
    for g in range(1, generations + 1):
        nxt = []
        for (o, sz, sgn, _gen, back) in frontier:
            for (nf, af) in fdata:
                nw = (nf[0] * sgn, nf[1] * sgn, nf[2] * sgn)
                if back is not None and (nw[0] * back[0]
                                         + nw[1] * back[1]
                                         + nw[2] * back[2]) < -0.999:
                    continue  # face occupied by the parent
                csz = sz * s
                d = af * (sz + csz)
                c = (o[0] + nw[0] * d, o[1] + nw[1] * d,
                     o[2] + nw[2] * d)
                k = key(c, csz)
                if k in seen:
                    continue
                seen.add(k)
                nxt.append((c, csz, -sgn if alternate else sgn, g, nw))
            if len(all_copies) + len(nxt) > MAX_COPIES:
                raise ValueError(
                    f"too many copies (>{MAX_COPIES}); "
                    "lower generations")
        frontier = nxt
        all_copies.extend(nxt)
    final = all_copies if keep_parents else (frontier or all_copies)
    verts = []
    faces = []
    face_gen = []
    for (o, sz, sgn, gen, _b) in final:
        base = len(verts)
        fs = sz * sgn
        for v in V:
            verts.append((o[0] + v[0] * fs, o[1] + v[1] * fs,
                          o[2] + v[2] * fs))
        for f in F:
            idx = [base + i for i in f]
            if sgn < 0:
                idx.reverse()
            faces.append(idx)
            face_gen.append(gen)
    return verts, faces, len(final), face_gen


def _convex_hull(pts):
    """Triangles of the 3D convex hull of `pts` (index triples,
    outward winding). Simple incremental algorithm -- adequate for
    the display shell it feeds."""
    n = len(pts)
    if n < 4:
        return []

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    diag = max(max(p[k] for p in pts) - min(p[k] for p in pts)
               for k in range(3)) or 1.0
    eps = 1e-7 * diag
    i0 = min(range(n), key=lambda i: pts[i])
    i1 = max(range(n), key=lambda i: pts[i])
    d01 = sub(pts[i1], pts[i0])

    def _dline(i):
        c = cross(d01, sub(pts[i], pts[i0]))
        return dot(c, c)

    i2 = max(range(n), key=_dline)
    if _dline(i2) <= eps * eps:
        return []
    pn = cross(d01, sub(pts[i2], pts[i0]))

    def _dplane(i):
        return abs(dot(pn, sub(pts[i], pts[i0])))

    i3 = max(range(n), key=_dplane)
    if _dplane(i3) <= eps * math.sqrt(dot(pn, pn)):
        return []
    inner = tuple(sum(pts[i][k] for i in (i0, i1, i2, i3)) / 4.0
                  for k in range(3))
    faces = {}
    nfid = [0]

    def add_face(a, b, c):
        nrm = _unit(cross(sub(pts[b], pts[a]), sub(pts[c], pts[a])))
        off = dot(nrm, pts[a])
        if dot(nrm, inner) > off:
            b, c = c, b
            nrm = tuple(-x for x in nrm)
            off = -off
        faces[nfid[0]] = (a, b, c, nrm, off)
        nfid[0] += 1

    for (a, b, c) in ((i0, i1, i2), (i0, i1, i3), (i0, i2, i3),
                      (i1, i2, i3)):
        add_face(a, b, c)
    order = sorted((i for i in range(n) if i not in (i0, i1, i2, i3)),
                   key=lambda i: -dot(sub(pts[i], inner),
                                      sub(pts[i], inner)))
    for ip in order:
        p = pts[ip]
        vis = [k for k, (a, b, c, nrm, off) in faces.items()
               if nrm[0] * p[0] + nrm[1] * p[1] + nrm[2] * p[2]
               - off > eps]
        if not vis:
            continue
        edges = set()
        for k in vis:
            a, b, c, _n, _o = faces[k]
            edges.update(((a, b), (b, c), (c, a)))
        horizon = [(u, v) for (u, v) in edges if (v, u) not in edges]
        for k in vis:
            del faces[k]
        for (u, v) in horizon:
            add_face(u, v, ip)
    return [(a, b, c) for (a, b, c, _n, _o) in faces.values()]


_HULL_CAP = 12000


def _hull_candidates_np(pts):
    """Exact reduction of a large point set to the subset spanning
    its convex hull, quickhull-style: hull an extreme seed set, then
    repeatedly add the points lying outside it (numpy for the bulk
    plane tests). Returns indices into pts, or None without numpy."""
    try:
        import numpy as np
    except ImportError:
        return None
    P = np.asarray(pts, dtype=np.float64)
    diag = float((P.max(axis=0) - P.min(axis=0)).max()) or 1.0
    eps = 1e-7 * diag
    dirs = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1)
                     for z in (-1, 0, 1) if x or y or z], float)
    dirs /= np.sqrt((dirs * dirs).sum(axis=1))[:, None]
    sel = set(np.argmax(P @ dirs.T, axis=0).tolist())
    sl = sorted(sel)
    for _ in range(64):
        sl = sorted(sel)
        sub = [pts[i] for i in sl]
        tris = _convex_hull(sub)
        if not tris:
            return None
        A = []
        b = []
        for (ia, ib, ic) in tris:
            pa, pb, pc = sub[ia], sub[ib], sub[ic]
            u = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
            w = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
            nr = _unit((u[1] * w[2] - u[2] * w[1],
                        u[2] * w[0] - u[0] * w[2],
                        u[0] * w[1] - u[1] * w[0]))
            A.append(nr)
            b.append(nr[0] * pa[0] + nr[1] * pa[1] + nr[2] * pa[2])
        A = np.array(A)
        bb = np.array(b)
        out = np.empty(len(P))
        for st in range(0, len(P), 8192):
            out[st:st + 8192] = (P[st:st + 8192] @ A.T
                                 - bb).max(axis=1)
        viol = [int(i) for i in np.nonzero(out > eps)[0]
                if int(i) not in sel]
        if not viol:
            return sl
        viol.sort(key=lambda i: -out[i])
        sel.update(viol[:512])
    return sl


def _hull_faces(verts, cap=_HULL_CAP):
    """Convex-hull triangles over the (deduplicated) mesh vertices.
    Oversized sets are reduced exactly via numpy when available;
    otherwise they are capped to the vertices emitted last -- the
    outermost generations -- as a best effort."""
    uniq = set()
    idxs = []
    for i in range(len(verts) - 1, -1, -1):
        v = verts[i]
        k = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        if k in uniq:
            continue
        uniq.add(k)
        idxs.append(i)
    if len(idxs) > cap:
        sel = _hull_candidates_np([verts[i] for i in idxs])
        if sel is not None:
            idxs = [idxs[j] for j in sel]
        else:
            idxs = idxs[:cap]
    pts = [verts[i] for i in idxs]
    return [[idxs[a], idxs[b], idxs[c]]
            for (a, b, c) in _convex_hull(pts)]


def _mat_mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def _mat_vec(A, v):
    return tuple(sum(A[i][k] * v[k] for k in range(3)) for i in range(3))


def _axis_rot(axis, ang):
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return ((c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C))


_ID3 = ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))


def _euler_rot(rx, ry, rz):
    R = _axis_rot((1, 0, 0), math.radians(rx))
    R = _mat_mul(_axis_rot((0, 1, 0), math.radians(ry)), R)
    return _mat_mul(_axis_rot((0, 0, 1), math.radians(rz)), R)


def build_fractal(kind='CUBE', mode='VERTS', generations=3,
                  child_scale=0.5, spread=1.27, keep_parents=True,
                  push=0.0, twist=0.0, rot_x=0.0, rot_y=0.0, rot_z=0.0,
                  scale=1.0, crystal='OFF', hull_shell=False):
    """Returns (verts, faces, n_copies, face_gen). Each copy carries
    (origin, size, orientation, generation). Children sit at the
    anchors of their parent (scaled by spread, shifted radially by
    push), rotated by `twist` about the anchor direction plus an
    XYZ rotation -- all cumulative across generations.

    With crystal != 'OFF' the Fathauer/Haüy face-growth engine is
    used instead (a preset key of _CRYSTAL_PRESETS, or 'CUSTOM' to
    apply the face rule to kind/child_scale); mode, spread, push,
    twist and the rotations are ignored there. hull_shell appends
    the convex hull of the cluster as extra faces tagged
    generation -1, so the crystal form reads clearly."""
    if crystal != 'OFF':
        verts, faces, ncop, face_gen = _build_crystal(
            crystal, kind, child_scale, generations, keep_parents)
        if hull_shell and verts:
            for tri in _hull_faces(verts):
                faces.append(tri)
                face_gen.append(-1)
        if scale != 1.0:
            verts = [(v[0] * scale, v[1] * scale, v[2] * scale)
                     for v in verts]
        return verts, faces, ncop, face_gen
    V, F = seed_poly(kind)
    anchors = _anchors(V, F, mode)
    Re = _euler_rot(rot_x, rot_y, rot_z)
    tw = math.radians(twist)
    copies = [((0.0, 0.0, 0.0), 1.0, _ID3, 0)]
    all_copies = list(copies) if keep_parents else []
    for g in range(generations):
        nxt = []
        for (o, s, R, _gen) in copies:
            for a in anchors:
                al = math.sqrt(sum(x * x for x in a)) or 1.0
                adir = tuple(x / al for x in a)
                dir_w = _mat_vec(R, adir)
                pos = tuple(o[k]
                            + _mat_vec(R, a)[k] * s * spread
                            + dir_w[k] * push * s for k in range(3))
                Rc = R
                if abs(tw) > 1e-12:
                    Rc = _mat_mul(_axis_rot(dir_w, tw), Rc)
                Rc = _mat_mul(Rc, Re)
                nxt.append((pos, s * child_scale, Rc, g + 1))
        copies = nxt
        if keep_parents:
            all_copies.extend(copies)
        if len(copies) * len(V) > MAX_COPIES * 4:
            raise ValueError(
                f"too many copies ({len(copies)}); lower generations")
    final = all_copies if keep_parents else copies
    if len(final) > MAX_COPIES:
        raise ValueError(f"too many copies ({len(final)})")
    verts = []
    faces = []
    face_gen = []
    for (o, s, R, gen) in final:
        base = len(verts)
        for v in V:
            p = _mat_vec(R, v)
            verts.append(tuple((o[k] + p[k] * s) * scale
                               for k in range(3)))
        for f in F:
            faces.append([base + i for i in f])
            face_gen.append(gen)
    if hull_shell and verts:
        for tri in _hull_faces(verts):
            faces.append(tri)
            face_gen.append(-1)
    return verts, faces, len(final), face_gen


try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, IntProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_fractal_poly_add(bpy.types.Operator):
        """Recursive Platonic solid cluster (vertex mode, scale 0.5,
        tetrahedron = the Sierpinski tetrahedron)"""
        bl_idname = "mesh.fractal_polyhedron_add"
        bl_label = "Fractal Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Solid",
            items=[('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", ""),
                   ('DODECA', "Dodecahedron", ""),
                   ('ICOSA', "Icosahedron", "")],
            default='CUBE')
        mode: EnumProperty(
            name="Anchors",
            items=[('VERTS', "Vertices", "children at vertices"),
                   ('FACES', "Faces", "children at face centres"),
                   ('EDGES', "Edges", "children at edge midpoints")],
            default='VERTS')
        generations: IntProperty(name="Generations", default=3,
                                 min=1, max=7)
        child_scale: FloatProperty(name="Child Scale", default=0.5,
                                   min=0.1, max=0.9)
        spread: FloatProperty(
            name="Spread", default=1.27, min=0.5, max=3.0,
            description="Distance multiplier from parent to children")
        keep_parents: BoolProperty(
            name="Keep Parents", default=True,
            description="Keep every generation (off = only the last, "
                        "e.g. the Sierpinski gasket)")
        push: FloatProperty(
            name="Push", default=0.0, min=-1.0, max=2.0,
            description="Extra shift of each child along its anchor "
                        "direction (in parent-size units)")
        twist: FloatProperty(
            name="Twist", default=0.0, min=-180.0, max=180.0,
            description="Rotation of each child about its anchor "
                        "direction (cumulative per generation)")
        rot_x: FloatProperty(name="Child Rotate X", default=0.0,
                             min=-180.0, max=180.0)
        rot_y: FloatProperty(name="Child Rotate Y", default=0.0,
                             min=-180.0, max=180.0)
        rot_z: FloatProperty(name="Child Rotate Z", default=0.0,
                             min=-180.0, max=180.0)
        crystal: EnumProperty(
            name="Crystal",
            description="Haüy/Fathauer fractal-crystal growth: stack "
                        "scaled copies face-to-face on every free "
                        "face of the previous generation (after "
                        "R.-J. Haüy 1784; R. Fathauer, Bridges "
                        "2008/2013)",
            items=[
                ('OFF', "Off",
                 "Classic vertex/face/edge cluster engine"),
                ('CUBE_OCT', "Cube s=1/2 (Octahedron)",
                 "Half-scale cubes; octahedral hull whose facets are "
                 "Sierpinski triangles (3 faces per hull vertex)"),
                ('CUBE_RD', "Cube s=1/3 (Rhombic Dodecahedron)",
                 "Third-scale cubes; outermost edges lie in the "
                 "planes of a rhombic dodecahedron"),
                ('HAUY_OCT', "Haüy Octahedron (s=1)",
                 "Equal cubes stacked in diminishing layers -- "
                 "Haüy's classic crystal construction (1784)"),
                ('DODECAFOAM', "Dodecafoam (s=phi^-2)",
                 "Dodecahedra at the inverse squared golden ratio "
                 "(~0.382), flush-fitting; organic, no simple hull "
                 "(subset of Goodman-Strauss dodecafoam)"),
                ('DODECA_ICOSA', "Dodecahedron s=1/2 (Icosahedron)",
                 "Half-scale dodecahedra; icosahedral hull with "
                 "Sierpinski-triangle facets"),
                ('ICOSA_SP', "Icosahedron s=phi^-2 (Sierpinski Pent)",
                 "Icosahedra at ~0.382; dodecahedral hull whose "
                 "facets are Sierpinski pentagons (5 faces per hull "
                 "vertex)"),
                ('HEX_DIPYR', "Hex Prism s=1/2 (Dipyramid)",
                 "Half-scale hexagonal prisms; hexagonal-dipyramid "
                 "hull with elongated Sierpinski triangles"),
                ('OCT_DIPYR', "Octagon Prism s=1/2 (Dipyramid)",
                 "Half-scale octagonal prisms; octagonal-dipyramid "
                 "hull"),
                ('CUSTOM', "Custom",
                 "Face-growth rule applied to the Solid and Child "
                 "Scale chosen above")],
            default='OFF')
        hull_shell: BoolProperty(
            name="Hull Shell", default=False,
            description="Add the convex hull of the cluster as a "
                        "translucent shell so the crystal form reads "
                        "clearly")
        coloring: EnumProperty(
            name="Coloring",
            items=[('GEN', "Per Generation",
                    "One material per generation (view with Material "
                    "Preview or Solid shading set to Material color)"),
                   ('NONE', "None", "No materials")],
            default='GEN')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = [(0.85, 0.82, 0.75), (0.90, 0.36, 0.23),
                    (0.27, 0.52, 0.79), (0.95, 0.77, 0.29),
                    (0.30, 0.69, 0.42), (0.62, 0.40, 0.75),
                    (0.25, 0.72, 0.72), (0.91, 0.56, 0.71)]

        @classmethod
        def _material_for(cls, g):
            if g < 0:
                name = "Fractal Hull Shell"
                mat = bpy.data.materials.get(name)
                if mat is None:
                    mat = bpy.data.materials.new(name)
                    mat.diffuse_color = (0.9, 0.95, 1.0, 0.25)
                    mat.use_nodes = True
                    bsdf = mat.node_tree.nodes.get("Principled BSDF")
                    if bsdf is not None:
                        bsdf.inputs["Base Color"].default_value = \
                            (0.9, 0.95, 1.0, 1.0)
                        bsdf.inputs["Alpha"].default_value = 0.25
                        bsdf.inputs["Roughness"].default_value = 0.3
                    if hasattr(mat, "surface_render_method"):
                        mat.surface_render_method = 'BLENDED'
                    elif hasattr(mat, "blend_method"):
                        mat.blend_method = 'BLEND'
                return mat
            name = f"Fractal Gen {g}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE[g % len(cls._PALETTE)]
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.55
            return mat

        def execute(self, context):
            try:
                verts, faces, nc, face_gen = build_fractal(
                    self.kind, self.mode, self.generations,
                    self.child_scale, self.spread, self.keep_parents,
                    self.push, self.twist, self.rot_x, self.rot_y,
                    self.rot_z, 1.0, self.crystal, self.hull_shell)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            if verts:
                xs = [v[0] for v in verts]
                ys = [v[1] for v in verts]
                zs = [v[2] for v in verts]
                cen = (0.5 * (min(xs) + max(xs)),
                       0.5 * (min(ys) + max(ys)),
                       0.5 * (min(zs) + max(zs)))
                ext = max(max(xs) - min(xs), max(ys) - min(ys),
                          max(zs) - min(zs))
                s = (2.0 * self.scale / ext) if ext > 1e-9 else 1.0
                verts = [((v[0] - cen[0]) * s,
                          (v[1] - cen[1]) * s,
                          (v[2] - cen[2]) * s) for v in verts]
            me = bpy.data.meshes.new("FractalPolyhedron")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if len(me.polygons) == len(faces):
                attr = me.attributes.new("generation", 'INT', 'FACE')
                attr.data.foreach_set('value', face_gen)
                if self.coloring == 'GEN':
                    gens = sorted(set(face_gen))
                    slot = {g: i for i, g in enumerate(gens)}
                    for g in gens:
                        me.materials.append(self._material_for(g))
                    me.polygons.foreach_set(
                        'material_index',
                        [slot[g] for g in face_gen])
            me.update()
            obj = bpy.data.objects.new("FractalPolyhedron", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{nc} copies")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'crystal')
            if self.crystal == 'OFF':
                for k in ('kind', 'mode', 'generations', 'child_scale',
                          'spread', 'keep_parents'):
                    lay.prop(self, k)
                col = lay.column(align=True)
                for k in ('push', 'twist', 'rot_x', 'rot_y', 'rot_z'):
                    col.prop(self, k)
            else:
                if self.crystal == 'CUSTOM':
                    lay.prop(self, 'kind')
                    lay.prop(self, 'child_scale')
                lay.prop(self, 'generations')
                lay.prop(self, 'keep_parents')
            lay.prop(self, 'hull_shell')
            lay.prop(self, 'coloring')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.fractal_polyhedron_add",
                             icon='OUTLINER_OB_POINTCLOUD')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_fractal_poly_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_fractal_poly_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for kind, mode, gen, cs, keep in (
                ('TETRA', 'VERTS', 3, 0.5, False),
                ('TETRA', 'VERTS', 4, 0.5, False),
                ('CUBE', 'FACES', 3, 0.45, True),
                ('ICOSA', 'VERTS', 2, 0.4, True)):
            v, f, n, fg = build_fractal(kind, mode, gen, cs, spread=1.0,
                                        keep_parents=keep, twist=15,
                                        rot_z=10)
            print(f"{kind} {mode} g{gen}: copies={n} verts={len(v)} "
                  f"gens={sorted(set(fg))}")
        for cr, gen in (('CUBE_OCT', 3), ('CUBE_RD', 3),
                        ('HAUY_OCT', 4), ('DODECAFOAM', 2),
                        ('DODECA_ICOSA', 2), ('ICOSA_SP', 2),
                        ('HEX_DIPYR', 3), ('OCT_DIPYR', 3)):
            v, f, n, fg = build_fractal(generations=gen, crystal=cr,
                                        hull_shell=True)
            print(f"crystal {cr} g{gen}: copies={n} verts={len(v)} "
                  f"faces={len(f)} gens={sorted(set(fg))}")
