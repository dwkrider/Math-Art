
# Rotegrity Generator for Blender
#
# Rotegrity ("nexorade") sphere models, after Antiprism's `rotegrity`:
# each edge of a spherical polyhedron becomes a curved strap on the
# sphere, rotated about its midpoint and lengthened so that strap ends
# overlap their neighbours in a woven, tensegrity-like arrangement.
# Twist and extension are free sliders (tune until the ends meet).
#
# References:
#   - Adrian Rossiter, Antiprism and its `rotegrity` program
#     (https://www.antiprism.com) -- the reference implementation.
#   - The term "rotegrity" was introduced by Richard (Dick) Boyt,
#     "Rotegrity" (1970); modern strap-sphere models by Dick
#     Fischbeck.
#   - "Tensegrity" is Buckminster Fuller's term for the floating-
#     compression structures of Kenneth Snelson.
#   - "Nexorade" / reciprocal-frame terminology after Olivier
#     Baverel et al., "Nexorades" (mutually supporting members).

bl_info = {
    "name": "Rotegrity Generator",
    "author": "Math Art project (after Antiprism's rotegrity)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Rotegrity",
    "description": "Rotegrity / nexorade strap-sphere models",
    "category": "Add Mesh",
}

import math
from math import sin, cos, pi

PHI = (1 + 5 ** 0.5) / 2


# ---- spherical seed polyhedra -------------------------------------------

def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


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
        # dual of icosahedron: face centers
        IV = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                IV += [(0, a, b), (a, b, 0), (b, 0, a)]
        IF = _icosa_faces(IV)
        V = []
        for f in IF:
            c = [sum(IV[i][k] for i in f) / 3 for k in range(3)]
            V.append(tuple(c))
        # faces: for each icosa vertex, its 5 adjacent icosa faces ordered
        F = []
        for vi in range(len(IV)):
            adj = [fi for fi, f in enumerate(IF) if vi in f]
            c = IV[vi]
            u = [V[adj[0]][k] - 0 for k in range(3)]
            n = _unit(c)
            u = [u[k] - sum(u[j] * n[j] for j in range(3)) * n[k]
                 for k in range(3)]
            w = (n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2],
                 n[0] * u[1] - n[1] * u[0])
            def ang(fi):
                d = V[fi]
                return math.atan2(sum(d[k] * w[k] for k in range(3)),
                                  sum(d[k] * u[k] for k in range(3)))
            F.append(sorted(adj, key=ang))
    else:
        raise ValueError(kind)
    return [_unit(v) for v in V], [list(f) for f in F]


def _icosa_faces(V):
    """Triangles of the icosahedron given its 12 vertices: all vertex
    triples at minimal mutual distance."""
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
    F = set()
    for i in range(n):
        for j in adj[i]:
            for k in adj[i] & adj[j]:
                F.add(tuple(sorted((i, j, k))))
    faces = []
    for f in F:
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


# ---- strap construction --------------------------------------------------

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


# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_rotegrity_add(bpy.types.Operator):
        """Add a rotegrity (nexorade) strap-sphere"""
        bl_idname = "mesh.rotegrity_add"
        bl_label = "Rotegrity"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Seed",
            items=[('ICOSA', "Icosahedron", ""), ('OCTA', "Octahedron", ""),
                   ('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('DODECA', "Dodecahedron", "")],
            default='ICOSA')
        freq: IntProperty(
            name="Geodesic Frequency", default=2, min=1, max=8,
            description="Subdivision of triangular seeds")
        twist: FloatProperty(
            name="Twist", default=18.0, min=-60.0, max=60.0,
            description="Rotation of each strap about its midpoint (deg)")
        extension: FloatProperty(
            name="Extension", default=0.35, min=0.0, max=1.0,
            description="Lengthening of each strap beyond its edge")
        width: FloatProperty(name="Strap Width", default=0.06,
                             min=0.005, max=0.4)
        thickness: FloatProperty(name="Strap Thickness", default=0.025,
                                 min=0.002, max=0.2)
        segments: IntProperty(name="Segments", default=12, min=4, max=64)
        coloring: EnumProperty(
            name="Coloring",
            items=[('LENGTH', "By Strap Length",
                    "One material per strap length class -- geodesic "
                    "breakdowns give a few distinct lengths, colored "
                    "as in physical rotegrity kits (view with "
                    "Material Preview or Solid shading set to "
                    "Material color)"),
                   ('STRAP', "Per Strap",
                    "Cycle the palette strap by strap"),
                   ('NONE', "None", "No materials")],
            default='LENGTH')
        scale: FloatProperty(name="Radius", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                    (0.45, 0.45, 0.85), (0.80, 0.50, 0.30)]

        @classmethod
        def _material_for(cls, i, kind="Strap"):
            name = f"Rotegrity {kind} {i + 1}"
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
            verts, faces, ne, face_strap, strap_len = build_rotegrity(
                self.kind, self.freq, self.twist, self.extension,
                self.width, self.thickness, self.segments, self.scale)
            me = bpy.data.meshes.new("Rotegrity")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            if len(me.polygons) == len(face_strap):
                classes = sorted(set(round(l, 5) for l in strap_len))
                strap_cls = [classes.index(round(l, 5))
                             for l in strap_len]
                attr = me.attributes.new("strap_index", 'INT', 'FACE')
                attr.data.foreach_set('value', face_strap)
                attr = me.attributes.new("length_class", 'INT',
                                         'FACE')
                attr.data.foreach_set(
                    'value', [strap_cls[s] for s in face_strap])
                if self.coloring == 'LENGTH':
                    for i in range(len(classes)):
                        me.materials.append(
                            self._material_for(i, "Length"))
                    me.polygons.foreach_set(
                        'material_index',
                        [strap_cls[s] for s in face_strap])
                elif self.coloring == 'STRAP':
                    nmat = min(ne, len(self._PALETTE))
                    for i in range(nmat):
                        me.materials.append(self._material_for(i))
                    me.polygons.foreach_set(
                        'material_index',
                        [s % nmat for s in face_strap])
            me.update()
            obj = bpy.data.objects.new("Rotegrity", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{ne} straps")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'freq', 'twist', 'extension', 'width',
                      'thickness', 'segments', 'coloring', 'scale'):
                if k == 'freq' and self.kind in ('CUBE', 'DODECA'):
                    continue
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.rotegrity_add", icon='SPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_rotegrity_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_rotegrity_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for kind, freq in (('ICOSA', 1), ('ICOSA', 2), ('CUBE', 1),
                           ('DODECA', 1)):
            v, f, ne, fs, sl = build_rotegrity(kind, freq)
            print(f"{kind} f{freq}: straps={ne} verts={len(v)} "
                  f"faces={len(f)}")
