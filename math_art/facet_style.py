
# Shared "Face Segments" style for the polyhedron generators.
#
# Dissect a convex polyhedron's shell into one segment per face: each
# face is extruded inward to a given depth, and every side wall lies
# on the plane through the shared edge whose normal bisects the two
# faces' outward normals -- a mitre cut at half the dihedral angle --
# so neighbouring segments share their bevel planes and meet flush.
# The outer cap is the original face; the inner cap sits `depth` below
# it.  Optionally each segment is exploded outward along its centroid
# direction, and/or emitted as a separate mesh object.
#
# This is the (correct, interlocking) descendant of a polyhedral shell
# dissection: the segments partition the solid exactly, verified by
# volume conservation at full depth.

from math import sqrt


def _face_normal(P):
    """Unit normal of the polygon P via Newell's method."""
    n = [0.0, 0.0, 0.0]
    m = len(P)
    for i in range(m):
        a, b = P[i], P[(i + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    L = sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) or 1.0
    return (n[0] / L, n[1] / L, n[2] / L)


def _det3(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    return a * (e * i - f * h) - b * (d * i - f * g) \
        + c * (d * h - e * g)


def _solve3(rows, rhs):
    """Solve rows . x = rhs by Cramer's rule; None if singular."""
    det = _det3(rows)
    if abs(det) < 1e-12:
        return None
    out = []
    for j in range(3):
        mm = [list(rows[r]) for r in range(3)]
        for r in range(3):
            mm[r][j] = rhs[r]
        out.append(_det3(mm) / det)
    return tuple(out)


def _orient_faces(verts, faces):
    """Wind every face outward from the (convex) part's centroid."""
    c = tuple(sum(v[k] for v in verts) / len(verts) for k in range(3))
    out = []
    for f in faces:
        p0, p1, p2 = verts[f[0]], verts[f[1]], verts[f[2]]
        u = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        v = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
        d = sum(n[k] * (p0[k] - c[k]) for k in range(3))
        out.append(f[::-1] if d < 0 else list(f))
    return out


def build_facets(V, F, depth=0.15, padding=0.0):
    """One inward-extruded, mitre-beveled segment per face of the
    convex polyhedron (V, F).  Returns a list of
    (verts, faces, n_sides, outward_dir)."""
    ctr = tuple(sum(v[k] for v in V) / len(V) for k in range(3))
    normals, offs = [], []
    for f in F:
        P = [V[i] for i in f]
        n = _face_normal(P)
        cen = tuple(sum(p[k] for p in P) / len(P) for k in range(3))
        if sum(n[k] * (cen[k] - ctr[k]) for k in range(3)) < 0:
            n = (-n[0], -n[1], -n[2])
        normals.append(n)
        offs.append(sum(n[k] * P[0][k] for k in range(3)))
    emap = {}
    for fi, f in enumerate(F):
        m = len(f)
        for k in range(m):
            emap.setdefault(frozenset((f[k], f[(k + 1) % m])),
                            []).append(fi)

    def unit_diff(na, nb):
        d = (na[0] - nb[0], na[1] - nb[1], na[2] - nb[2])
        L = sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) or 1.0
        return (d[0] / L, d[1] / L, d[2] / L)

    segs = []
    for fi, f in enumerate(F):
        n_i, d_i, m = normals[fi], offs[fi], len(f)
        cen = tuple(sum(V[i][k] for i in f) / m for k in range(3))
        outer = [V[i] for i in f]
        inner = []
        for k in range(m):
            v, vp, vn = f[k], f[(k - 1) % m], f[(k + 1) % m]
            jA = [x for x in emap[frozenset((vp, v))] if x != fi]
            jB = [x for x in emap[frozenset((v, vn))] if x != fi]
            nA = normals[jA[0]] if jA else n_i
            nB = normals[jB[0]] if jB else n_i
            mA = unit_diff(n_i, nA)
            mB = unit_diff(n_i, nB)
            Vv = V[v]
            cA = sum(mA[k] * Vv[k] for k in range(3)) + padding
            cB = sum(mB[k] * Vv[k] for k in range(3)) + padding
            pt = _solve3((mA, mB, n_i), (cA, cB, d_i - depth))
            if pt is None:
                pt = tuple(cen[k] + (Vv[k] - cen[k]) * 0.3
                           for k in range(3))
            inner.append(pt)
        verts = list(outer) + inner
        faces = []
        for k in range(m):
            j = (k + 1) % m
            faces.append([k, j, m + j, m + k])       # side wall
        faces.append(list(range(m)))                 # outer cap
        faces.append([m + k for k in range(m)])       # inner cap
        faces = _orient_faces(verts, faces)
        L = sqrt(sum(c * c for c in cen)) or 1.0
        segs.append((verts, faces, m, tuple(c / L for c in cen)))
    return segs


try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _facet_mesh(name, verts, faces, sizes, material_fn):
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        if len(me.polygons) == len(sizes):
            attr = me.attributes.new("ngon_sides", 'INT', 'FACE')
            attr.data.foreach_set('value', sizes)
            if material_fn is not None:
                lut = {}
                for nn in sorted(set(sizes)):
                    mat = material_fn(nn)
                    if mat is not None:
                        lut[nn] = len(me.materials)
                        me.materials.append(mat)
                if lut:
                    me.polygons.foreach_set(
                        'material_index',
                        [lut.get(s, 0) for s in sizes])
        me.update()
        return me

    def emit_facets(context, V, F, label, depth=0.15, padding=0.0,
                    explode=0.1, separate=False, material_fn=None):
        """Build the Face Segments of the convex polyhedron (V, F) and
        add them to the scene.  Returns (first_object, n_segments).
        `material_fn(n_sides)` may return a material per face size, or
        None for no colouring."""
        segs = build_facets(V, F, depth, padding)
        for o in context.selected_objects:
            o.select_set(False)
        cur = context.scene.cursor.location
        first = None
        if separate:
            for k, (verts, faces, ns, d) in enumerate(segs):
                nm = f"{label} facet {k + 1}of{len(segs)}"
                me = _facet_mesh(nm, verts, faces, [ns] * len(faces),
                                 material_fn)
                obj = bpy.data.objects.new(nm, me)
                context.collection.objects.link(obj)
                obj.location = (cur[0] + explode * d[0],
                                cur[1] + explode * d[1],
                                cur[2] + explode * d[2])
                obj.select_set(True)
                if first is None:
                    first = obj
        else:
            av, af, sizes = [], [], []
            for verts, faces, ns, d in segs:
                base = len(av)
                ox, oy, oz = (explode * d[0], explode * d[1],
                              explode * d[2])
                av.extend((v[0] + ox, v[1] + oy, v[2] + oz)
                          for v in verts)
                af.extend([base + i for i in f] for f in faces)
                sizes.extend([ns] * len(faces))
            me = _facet_mesh(f"{label} facets", av, af, sizes,
                             material_fn)
            obj = bpy.data.objects.new(f"{label} facets", me)
            context.collection.objects.link(obj)
            obj.location = cur
            obj.select_set(True)
            first = obj
        context.view_layer.objects.active = first
        return first, len(segs)

    def register():
        pass

    def unregister():
        pass
