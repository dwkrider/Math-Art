# Weaves over stellated forms.
#
# Part of the Math Art weaving engine (`math_art/weaving/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#

import math
from math import sqrt


PHI = (sqrt(5) + 1) / 2


INV_PHI = (sqrt(5) - 1) / 2


FACE_STAR_RATIO = (5 - sqrt(5)) / 10


BEND_MITER_EXT = sqrt((5 + sqrt(5)) / 10)


BEND_INNER_EXT = sqrt((5 - sqrt(5)) / 10)


DODECA_VERTICES = [
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    (0, PHI, INV_PHI), (INV_PHI, 0, PHI), (PHI, INV_PHI, 0),
    (0, PHI, -INV_PHI), (-INV_PHI, 0, PHI), (PHI, -INV_PHI, 0),
    (0, -PHI, INV_PHI), (INV_PHI, 0, -PHI), (-PHI, INV_PHI, 0),
    (0, -PHI, -INV_PHI), (-INV_PHI, 0, -PHI), (-PHI, -INV_PHI, 0),
]


DODECA_FACES = [
    [0, 17, 14, 1, 19], [0, 19, 16, 2, 18], [0, 18, 15, 4, 17],
    [1, 12, 3, 16, 19], [1, 14, 5, 9, 12], [2, 16, 3, 8, 11],
    [2, 11, 6, 15, 18], [3, 12, 9, 7, 8], [4, 15, 6, 10, 13],
    [4, 13, 5, 14, 17], [5, 13, 10, 7, 9], [6, 11, 8, 7, 10],
]


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mul(s, p):
    return (s * p[0], s * p[1], s * p[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _len(p):
    return sqrt(_dot(p, p))


def _unit(p):
    l = _len(p)
    return _mul(1.0 / l, p)


def _avg(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n,
            sum(p[2] for p in pts) / n)


def _toward(a, b, t):
    return _add(a, _mul(t, _sub(b, a)))


def _reject(v, axis):
    return _sub(v, _mul(_dot(v, axis), axis))


def _isect(oa, da, ob, db):
    """Point on line A closest to line B (mitre intersection)."""
    w = _sub(ob, oa)
    aa = _dot(da, da)
    bb = _dot(db, db)
    ab = _dot(da, db)
    den = aa * bb - ab * ab
    if abs(den) < 1e-10:
        return _avg([oa, ob])
    t = (_dot(w, da) * bb - _dot(w, db) * ab) / den
    return _add(oa, _mul(t, da))


def _sort_by_direction(items, axis):
    """items: list of (obj, direction); sorted by angle about axis."""
    first = _reject(items[0][1], axis)
    u = _unit(first)
    v = _cross(axis, u)

    def ang(it):
        d = it[1]
        return math.atan2(_dot(d, v), _dot(d, u))
    return [it[0] for it in sorted(items, key=ang)]


class _Face:
    __slots__ = ('index', 'center', 'normal', 'vertexIndices',
                 'vertices')


def indexed_ssd_faces():
    """The 12 pentagram faces of the small stellated dodecahedron:
    per dodeca face, the apexes of its five neighbours in star
    order."""
    centers = [_avg([DODECA_VERTICES[i] for i in f])
               for f in DODECA_FACES]
    normals = []
    for f, c in zip(DODECA_FACES, centers):
        a, b, d = (DODECA_VERTICES[i] for i in f[:3])
        n = _unit(_cross(_sub(b, a), _sub(d, a)))
        if _dot(n, c) < 0:
            n = _mul(-1, n)
        normals.append(n)
    apex_h = sqrt(2 + 2 / sqrt(5))
    ssd_verts = [_add(c, _mul(apex_h, n))
                 for c, n in zip(centers, normals)]
    # adjacency: faces sharing an edge
    sets = [set(f) for f in DODECA_FACES]
    adj = [set() for _ in DODECA_FACES]
    for i in range(12):
        for j in range(i + 1, 12):
            if len(sets[i] & sets[j]) == 2:
                adj[i].add(j)
                adj[j].add(i)
    faces = []
    for fi, f in enumerate(DODECA_FACES):
        c = centers[fi]
        n = normals[fi]
        u = _unit(_sub(DODECA_VERTICES[f[0]], c))
        v = _cross(n, u)

        def angof(k):
            d = _sub(ssd_verts[k], c)
            return math.atan2(_dot(d, v), _dot(d, u))
        cyc = sorted(adj[fi], key=angof)
        idxs = [cyc[(2 * i) % 5] for i in range(5)]
        fc = _Face()
        fc.index = fi
        fc.vertexIndices = idxs
        fc.vertices = [ssd_verts[k] for k in idxs]
        fc.center = _avg(fc.vertices)
        fc.normal = n
        faces.append(fc)
    return faces


class _Arm:
    __slots__ = ('face', 'localIndex', 'tipIndex', 'points')


def build_arms(faces):
    arms = []
    for face in faces:
        for li in range(5):
            a = _Arm()
            a.face = face
            a.localIndex = li
            a.tipIndex = face.vertexIndices[li]
            start = face.vertices[li]
            bend = _toward(start, face.vertices[(li + 1) % 5],
                           FACE_STAR_RATIO)
            a.points = [start, bend, _avg(face.vertices)]
            arms.append(a)
    return arms


def _face_inward(face, apex, edge_dir):
    return _unit(_reject(_sub(face.center, apex), edge_dir))


def _edge_face_map(faces):
    ef = {}
    for face in faces:
        for i in range(5):
            a = face.vertexIndices[i]
            b = face.vertexIndices[(i + 1) % 5]
            ef.setdefault((min(a, b), max(a, b)), []).append(face)
    return ef


def _miter_point_in_face(face, tip_index, width):
    li = face.vertexIndices.index(tip_index)
    apex = face.vertices[li]
    pd = _unit(_sub(face.vertices[(li + 4) % 5], apex))
    nd = _unit(_sub(face.vertices[(li + 1) % 5], apex))
    pi_ = _face_inward(face, apex, pd)
    ni = _face_inward(face, apex, nd)
    return _isect(_add(apex, _mul(width, pi_)), pd,
                  _add(apex, _mul(width, ni)), nd)


class _Geo:
    __slots__ = ('arm', 'otherFace', 'outerEnd', 'innerApex',
                 'firstStart', 'firstEnd', 'firstFace',
                 'secondStart', 'secondEnd', 'secondFace')


def _arm_geometry(arm, other, face_miters, inner_apex, width):
    apex, bend = arm.points[0], arm.points[1]
    ed = _unit(_sub(bend, apex))
    g = _Geo()
    g.arm = arm
    g.otherFace = other
    g.outerEnd = bend
    g.innerApex = inner_apex
    g.firstFace = arm.face
    g.firstStart = face_miters[arm.face.index]
    g.firstEnd = _add(bend, _mul(width,
                                 _face_inward(arm.face, apex, ed)))
    g.secondFace = other
    g.secondStart = face_miters[other.index]
    g.secondEnd = _add(bend, _mul(width,
                                  _face_inward(other, apex, ed)))
    return g


def _edge_dir(g):
    return _unit(_sub(g.outerEnd, g.arm.points[0]))


def _miter_width(g):
    return _len(_sub(g.firstEnd, g.outerEnd))


def _m_outer(g):
    return _add(g.outerEnd,
                _mul(BEND_MITER_EXT * _miter_width(g), _edge_dir(g)))


def _m_first(g):
    return _add(g.firstEnd,
                _mul(BEND_INNER_EXT * _miter_width(g), _edge_dir(g)))


def _m_second(g):
    return _add(g.secondEnd,
                _mul(BEND_INNER_EXT * _miter_width(g), _edge_dir(g)))


def _m_par_inner(g):
    adj = _add(g.secondEnd, _sub(g.firstEnd, g.outerEnd))
    return _add(adj, _mul((2 * BEND_INNER_EXT - BEND_MITER_EXT)
                          * _miter_width(g), _edge_dir(g)))


def _m_side_in_face(g, face):
    if g.firstFace.index == face.index:
        return _m_first(g)
    if g.secondFace.index == face.index:
        return _m_second(g)
    raise ValueError("geometry not incident to face")


def _star_patch(face, tip_index, geometries, face_miters):
    li = face.vertexIndices.index(tip_index)
    apex = face.vertices[li]
    pd = _unit(_sub(face.vertices[(li + 4) % 5], apex))
    nd = _unit(_sub(face.vertices[(li + 1) % 5], apex))
    prev_g = next_g = None
    for g in geometries:
        if face.index not in (g.firstFace.index, g.secondFace.index):
            continue
        d = _unit(_sub(_m_outer(g), apex))
        if _dot(d, pd) > _dot(d, nd):
            prev_g = g
        else:
            next_g = g
    return [apex, _m_outer(prev_g), _m_side_in_face(prev_g, face),
            face_miters[face.index], _m_side_in_face(next_g, face),
            _m_outer(next_g)]


def _tip_polygons(faces, arms, edge_faces, tip_index, width):
    selected = [a for a in arms if a.tipIndex == tip_index]
    apex = selected[0].points[0]
    ordered_arms = _sort_by_direction(
        [(a, _unit(_sub(a.points[1], a.points[0]))) for a in selected],
        _unit(apex))
    incident = [f for f in faces if tip_index in f.vertexIndices]
    face_miters = {f.index: _miter_point_in_face(f, tip_index, width)
                   for f in incident}
    inner_apex = _add(apex, _mul(2 * width, _unit(_mul(-1, apex))))
    geometries = {}
    ordered = []
    for arm in ordered_arms:
        next_tip = arm.face.vertexIndices[(arm.localIndex + 1) % 5]
        owners = edge_faces[(min(tip_index, next_tip),
                             max(tip_index, next_tip))]
        other = owners[1] if owners[0].index == arm.face.index \
            else owners[0]
        g = _arm_geometry(arm, other, face_miters, inner_apex, width)
        geometries[(arm.face.index, arm.localIndex)] = g
        ordered.append(g)
    polys = []
    tags = []
    for f in incident:
        polys.append(_star_patch(f, tip_index, ordered, face_miters))
        tags.append(f.index)
    for g in ordered:
        inner_end = _m_par_inner(g)
        polys.append([g.firstStart, _m_first(g), inner_end,
                      g.innerApex])
        tags.append(g.firstFace.index)
        polys.append([g.secondStart, g.innerApex, inner_end,
                      _m_second(g)])
        tags.append(g.secondFace.index)
    return geometries, polys, tags


def _face_center_polygons(face, geometries):
    entries = []
    for li in range(5):
        g = geometries[(face.index, li)]
        entries.append((g, _unit(_sub(_m_outer(g), face.center))))
    ordered = _sort_by_direction(entries, face.normal)
    tangents = [_unit(_sub(face.center, g.outerEnd)) for g in ordered]
    left = [_m_outer(g) for g in ordered]
    right = [_m_first(g) for g in ordered]
    lower_left = [_m_second(g) for g in ordered]
    lower_right = [_m_par_inner(g) for g in ordered]

    def center_miters(before, after):
        out = []
        n = len(before)
        for i in range(n):
            p = (i + n - 1) % n
            out.append(_isect(before[i], tangents[i],
                              after[p], tangents[p]))
        return out

    top = center_miters(right, left)
    bottom = center_miters(lower_right, lower_left)
    polys = []
    n = len(ordered)
    for i in range(n):
        j = (i + 1) % n
        polys.append([left[i], right[i], top[i], top[j]])
        polys.append([lower_left[i], bottom[j], bottom[i],
                      lower_right[i]])
        polys.append([right[i], lower_right[i], bottom[i], top[i]])
        polys.append([left[i], top[j], bottom[j], lower_left[i]])
    polys.append(list(top))
    polys.append(list(reversed(bottom)))
    return polys


def build_weave(width=0.12, scale=1.0):
    """All polygons of the stellated surface weave, tagged with the
    pentagram-plane (strip) index. Returns (verts, faces_ix, tags)
    with welded vertices."""
    faces = indexed_ssd_faces()
    arms = build_arms(faces)
    edge_faces = _edge_face_map(faces)
    geometries = {}
    polys = []
    tags = []
    tips = sorted(set(a.tipIndex for a in arms))
    for tip in tips:
        g, p, t = _tip_polygons(faces, arms, edge_faces, tip, width)
        geometries.update(g)
        polys.extend(p)
        tags.extend(t)
    for face in faces:
        fp = _face_center_polygons(face, geometries)
        polys.extend(fp)
        tags.extend([face.index] * len(fp))
    # weld
    s = scale / sqrt(3.0)          # dodeca verts at radius sqrt(3)
    verts = []
    vid = {}
    faces_ix = []
    for poly in polys:
        ix = []
        for p in poly:
            key = (round(p[0], 8), round(p[1], 8), round(p[2], 8))
            i = vid.get(key)
            if i is None:
                i = len(verts)
                vid[key] = i
                verts.append((p[0] * s, p[1] * s, p[2] * s))
            ix.append(i)
        dedup = [ix[k] for k in range(len(ix))
                 if ix[k] != ix[k - 1]]
        if len(dedup) >= 3:
            faces_ix.append(dedup)
    return verts, faces_ix, tags
