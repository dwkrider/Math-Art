
# Stellated Surface Weave for Blender
#
# A port of Shengyi Wang's "Stellated Surface Weave"
# (txyyss.github.io/math-art/stellated-surface-weave, source at
# github.com/txyyss/math-art): twelve folded strips interlocking
# along the pentagram planes of a small stellated dodecahedron.
#
# Construction: each pentagram face carries five skeleton arms
# (tip -> bend at (5-sqrt5)/10 along the edge -> face centre). At
# every tip, the five incident arms share a pentagonal-pyramid apex
# whose inner point sits 2*width inward; strips get two side faces
# lying in the adjacent star planes plus a parallelogram closure.
# Toward the face centres the strips stay parallel and meet in
# mitred pentagon junctions. The two extension factors
# sqrt((5+sqrt5)/10) and sqrt((5-sqrt5)/10) keep the bends mitred at
# constant width. The strip width is the one free parameter.

bl_info = {
    "name": "Stellated Surface Weave",
    "author": "Math Art project (after Shengyi Wang / "
              "txyyss)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Stellated Surface Weave",
    "description": "Interlocking folded strips on the small "
                   "stellated dodecahedron",
    "category": "Add Mesh",
}

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


try:
    import bpy
    import bmesh
    from bpy.props import (FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_stellated_weave_add(bpy.types.Operator):
        """Twelve interlocking folded strips along the pentagram
        planes of a small stellated dodecahedron (after Shengyi
        Wang's Stellated Surface Weave)"""
        bl_idname = "mesh.stellated_weave_add"
        bl_label = "Stellated Surface Weave"
        bl_options = {'REGISTER', 'UNDO'}

        width: FloatProperty(
            name="Strip Width", default=0.12, min=0.02, max=0.32,
            description="Width of the folded strips; all mitres "
                        "recompute")
        coloring: EnumProperty(
            name="Coloring",
            items=[('STRIP', "Per Strip",
                    "One material per pentagram plane (12 strips; "
                    "view with Material Preview or Solid shading "
                    "set to Material colour)"),
                   ('NONE', "None", "No materials")],
            default='STRIP')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                    (0.45, 0.45, 0.85), (0.80, 0.50, 0.30),
                    (0.35, 0.60, 0.55), (0.75, 0.35, 0.45)]

        @classmethod
        def _material_for(cls, i):
            name = f"Weave Strand {i + 1}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE[i % len(cls._PALETTE)]
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb,
                                                               1.0)
                    bsdf.inputs["Roughness"].default_value = 0.45
            return mat

        def execute(self, context):
            verts, faces, tags = build_weave(self.width, self.scale)
            me = bpy.data.meshes.new("StellatedWeave")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v.co[k] for v in me.vertices)
                  for k in range(3)]
            hi = [max(v.co[k] for v in me.vertices)
                  for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0
                       for k in range(3)) or 1.0
            f = self.scale / half
            for v in me.vertices:
                v.co = [(v.co[k] - (lo[k] + hi[k]) / 2.0) * f
                        for k in range(3)]
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            if len(me.polygons) == len(tags):
                attr = me.attributes.new("strip_index", 'INT',
                                         'FACE')
                attr.data.foreach_set('value', tags)
                if self.coloring == 'STRIP':
                    for i in range(12):
                        me.materials.append(self._material_for(i))
                    me.polygons.foreach_set('material_index', tags)
            me.update()
            obj = bpy.data.objects.new("StellatedWeave", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('width', 'coloring', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.stellated_weave_add",
                             icon='MOD_LATTICE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_stellated_weave_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_stellated_weave_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        from collections import Counter
        faces = indexed_ssd_faces()
        # 12 pentagram faces from 12 tips, 5 arms each
        arms = build_arms(faces)
        tips = set(a.tipIndex for a in arms)
        print(f"faces={len(faces)}(12) tips={len(tips)}(12) "
              f"arms={len(arms)}(60) "
              f"{'OK' if (len(faces), len(tips), len(arms)) == (12, 12, 60) else 'BAD'}")
        for w in (0.06, 0.12, 0.2):
            v, f, t = build_weave(w)
            cnt = Counter()
            for fc in f:
                for i in range(len(fc)):
                    a, b = fc[i], fc[(i + 1) % len(fc)]
                    cnt[(min(a, b), max(a, b))] += 1
            nonman = sum(1 for c in cnt.values() if c != 2)
            print(f"width {w}: verts={len(v)} faces={len(f)} "
                  f"strips={len(set(t))}(12) "
                  f"non-2-manifold edges={nonman}")
