
# Regular Polylinks Generator for Blender
#
# Orderly tangles of regular polygons, after George W. Hart's regular
# polylinks: one polygonal frame per face (plane) of a Platonic solid,
# each rotated about its face normal, scaled, and offset so the frames
# interlock. Classic examples: 4 triangles (tetrahedron), 6 squares
# (cube), 12 pentagons or 6 pentagons (dodecahedron), 8 triangles
# (octahedron), 20 triangles (icosahedron).
#
# References:
#   - Alan Holden, "Orderly Tangles: Cloverleafs, Gordian Knots, and
#     Regular Polylinks" (Columbia University Press, 1983) -- the
#     symmetric interlocked-polygon arrangements.
#   - George W. Hart, "Orderly Tangles Revisited",
#     https://www.georgehart.com/orderly-tangles-revisited/tangles.htm
#     (the regular polylinks this follows).

bl_info = {
    "name": "Regular Polylinks",
    "author": "Math Art project (after George W. Hart)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Regular Polylinks",
    "description": "Interlocked polygon frames from Platonic solids",
    "category": "Add Mesh",
}

import math

import numpy as np

try:                                  # inside the math_art package
    from .curve_frames import closed_tube as _shared_closed_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import closed_tube as _shared_closed_tube
from math import cos, sin, pi

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


try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    PRESETS = {
        'T4': ("4 Triangles", dict(kind='TETRA', size=1.55, rotation=30,
                                   offset=-0.45, antipodal=False)),
        'C6': ("6 Squares", dict(kind='CUBE', size=1.45, rotation=25,
                                 offset=-0.4, antipodal=False)),
        'O8': ("8 Triangles", dict(kind='OCTA', size=1.6, rotation=25,
                                   offset=-0.25, antipodal=False)),
        'D6': ("6 Pentagons", dict(kind='DODECA', size=1.5, rotation=20,
                                   offset=-0.9, antipodal=True)),
        'D12': ("12 Pentagons", dict(kind='DODECA', size=1.25, rotation=15,
                                     offset=-0.5, antipodal=False)),
        'I20': ("20 Triangles", dict(kind='ICOSA', size=1.55, rotation=22,
                                     offset=-0.55, antipodal=False)),
    }

    class MESH_OT_polylinks_add(bpy.types.Operator):
        """Interlocked regular polygon frames (after Hart's polylinks)"""
        bl_idname = "mesh.polylinks_add"
        bl_label = "Regular Polylinks"
        bl_options = {'REGISTER', 'UNDO'}

        def _preset_chosen(self, context):
            if self.preset != 'CUSTOM':
                for k, v in PRESETS[self.preset][1].items():
                    setattr(self, k, v)

        preset: EnumProperty(
            name="Preset",
            items=[('CUSTOM', "Custom", "")] +
                  [(k, v[0], "") for k, v in PRESETS.items()],
            default='T4', update=_preset_chosen)
        kind: EnumProperty(
            name="Solid",
            items=[('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", ""),
                   ('DODECA', "Dodecahedron", ""),
                   ('ICOSA', "Icosahedron", "")],
            default='TETRA')
        size: FloatProperty(name="Frame Size", default=1.55,
                            min=0.5, max=4.0)
        rotation: FloatProperty(
            name="Rotation", default=30.0, min=-90.0, max=90.0,
            description="Turn of each frame about its face normal (deg)")
        offset: FloatProperty(
            name="Plane Offset", default=-0.45, min=-2.0, max=2.0,
            description="Push of each frame along its normal "
                        "(negative = toward the centre)")
        link_shape: EnumProperty(
            name="Link Shape",
            items=[('POLYGON', "Polygon Frame",
                    "Flat polygon frames (Hart's polylinks)"),
                   ('WAVE', "Wavy Circle",
                    "Radius-modulated circles: round rings that "
                    "weave through each other (after Shengyi "
                    "Wang's polylink add-on)"),
                   ('KNOT', "Torus Knot",
                    "A (p, q x sides) torus knot about each face "
                    "axis, swept with rotation-minimizing frames")],
            default='POLYGON')
        amplitude: FloatProperty(
            name="Wave Amplitude", default=0.35, min=0.0, max=2.0,
            description="Radial wave amplitude (wavy circle) or "
                        "knot minor radius")
        wave_factor: bpy.props.IntProperty(
            name="Wave Factor", default=1, min=1, max=8,
            description="Wave frequency in multiples of the face "
                        "side count")
        knot_p: bpy.props.IntProperty(
            name="Knot p", default=2, min=1, max=8,
            description="Windings around the face axis")
        knot_q_factor: bpy.props.IntProperty(
            name="Knot q Factor", default=1, min=1, max=8,
            description="q = factor x face side count")
        tube_sides: bpy.props.IntProperty(name="Tube Sides",
                                          default=8, min=3, max=24)
        segments: bpy.props.IntProperty(
            name="Link Segments", default=128, min=24, max=512,
            description="Samples along wavy / knot centerlines")
        width: FloatProperty(name="Frame Width", default=0.14,
                             min=0.02, max=0.9)
        thickness: FloatProperty(name="Frame Thickness", default=0.10,
                                 min=0.01, max=1.0)
        antipodal: BoolProperty(
            name="Antipodal Half", default=False,
            description="Use only one face of each antipodal pair")
        coloring: EnumProperty(
            name="Coloring",
            items=[('FRAME', "Per Link", "One material per frame, for "
                    "visibility (view with Material Preview or Solid "
                    "shading set to Material color)"),
                   ('PAIR', "Per Parallel Pair",
                    "Iso-color parallel (antipodal) frames, as in "
                    "Hart's paper models: 6 squares in 3 colors"),
                   ('NONE', "None", "No materials")],
            default='FRAME')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                    (0.80, 0.50, 0.30), (0.45, 0.45, 0.85)]

        @classmethod
        def _material_for(cls, i):
            name = f"Polylink {i + 1}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                if i < len(cls._PALETTE):
                    rgb = cls._PALETTE[i]
                else:
                    import colorsys
                    rgb = colorsys.hsv_to_rgb((i * 0.618034) % 1.0,
                                              0.6, 0.8)
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.5
            return mat

        def execute(self, context):
            if self.preset != 'CUSTOM':
                self._preset_chosen(context)
            verts, faces, nf, face_frame, frame_dirs = build_polylinks(
                self.kind, self.size, self.rotation, self.offset,
                self.width, self.thickness, self.antipodal, self.scale,
                self.link_shape, self.amplitude, self.wave_factor,
                self.knot_p, self.knot_q_factor, self.tube_sides,
                self.segments)
            me = bpy.data.meshes.new("Polylinks")
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
            if len(me.polygons) == len(faces):
                # frame membership as face metadata (Geometry Nodes /
                # shader Attribute node: "link_index")
                attr = me.attributes.new("link_index", 'INT', 'FACE')
                attr.data.foreach_set('value', face_frame)
                if self.coloring != 'NONE':
                    if self.coloring == 'PAIR':
                        group = {}
                        gid = []
                        for d in frame_dirs:
                            dd = d if tuple(d) >= tuple(-x for x in d) \
                                else tuple(-x for x in d)
                            key = tuple(round(x, 5) for x in dd)
                            if key not in group:
                                group[key] = len(group)
                            gid.append(group[key])
                    else:
                        gid = list(range(nf))
                    nmats = max(gid) + 1
                    for i in range(nmats):
                        me.materials.append(self._material_for(i))
                    me.polygons.foreach_set(
                        'material_index',
                        [gid[face_frame[i]] for i in range(len(faces))])
            me.update()
            obj = bpy.data.objects.new("Polylinks", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{nf} frames")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'link_shape')
            keys = ['kind', 'size', 'rotation', 'offset']
            if self.link_shape == 'POLYGON':
                keys += ['width', 'thickness']
            else:
                keys += ['thickness', 'amplitude']
                if self.link_shape == 'WAVE':
                    keys += ['wave_factor']
                else:
                    keys += ['knot_p', 'knot_q_factor']
                keys += ['tube_sides', 'segments']
            keys += ['antipodal', 'coloring', 'scale']
            for k in keys:
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.polylinks_add", icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_polylinks_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polylinks_add)
